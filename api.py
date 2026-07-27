# api.py
from __future__ import annotations
import os
import logging
import traceback
from io import BytesIO
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# pipeline imports (must be importable from repo root)
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

# Optional ML predict wrapper (if present)
try:
    from preprocessing.ml_categorizer import predict_categories  # type: ignore
    PREDICT_AVAILABLE = True
except Exception:
    PREDICT_AVAILABLE = False

# joblib for loading model
from joblib import load

logger = logging.getLogger("prosperity_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="ProsperityScore API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global holder for loaded ML model (pipeline)
ML_MODEL = None


def _download_s3_to_local(s3_path: str, local_path: str):
    """Download s3://bucket/key to local_path using boto3 (requires AWS env vars)."""
    import boto3

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )
    assert s3_path.startswith("s3://")
    _, rest = s3_path.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    logger.info("Downloading model from S3: s3://%s/%s -> %s", bucket, key, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, local_path)


@app.on_event("startup")
def startup_load_model():
    """Load ML model at startup if MODEL_PATH env var is set. Supports local path or s3:// URI."""
    global ML_MODEL
    model_path = os.environ.get("MODEL_PATH", "models/ml_model.pkl")
    try:
        if model_path.startswith("s3://"):
            local_tmp = "/tmp/ml_model.pkl"
            _download_s3_to_local(model_path, local_tmp)
            ML_MODEL = load(local_tmp)
            logger.info("Loaded model from S3 -> %s", local_tmp)
        else:
            p = Path(model_path)
            if p.exists():
                ML_MODEL = load(str(p))
                logger.info("Loaded model from %s", p)
            else:
                ML_MODEL = None
                logger.warning("MODEL_PATH %s not found at startup; ML categorizer unavailable", model_path)
    except Exception as e:
        logger.exception("Failed to load ML model at startup: %s", e)
        ML_MODEL = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ML_MODEL is not None}


def _read_uploaded_file(upload_file: UploadFile) -> pd.DataFrame:
    content = upload_file.file.read()
    upload_file.file.close()
    bio = BytesIO(content)
    fname = (upload_file.filename or "").lower()
    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(bio)
        elif fname.endswith(".csv"):
            try:
                bio.seek(0)
                df = pd.read_csv(bio)
            except Exception:
                bio.seek(0)
                df = pd.read_csv(bio, encoding="latin1")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Provide .csv, .xls or .xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")
    return df


def _pick_top_features(component_scores: Dict[str, float], raw: Dict[str, Any], top_k: int = 2) -> List[Dict[str, Any]]:
    # same heuristic as before
    comps_sorted = sorted(component_scores.items(), key=lambda kv: abs(kv[1] - 50), reverse=True)
    selected = []
    for comp_name, comp_val in comps_sorted[:top_k]:
        comp_raw = raw.get(f"{comp_name}_raw") or raw.get(comp_name) or {}
        if not isinstance(comp_raw, dict):
            try:
                comp_raw = getattr(comp_raw, "__dict__", dict(comp_raw))
            except Exception:
                comp_raw = dict()
        score_keys = [k for k in comp_raw.keys() if k.endswith("_score")]
        if score_keys:
            worst_key = min(score_keys, key=lambda k: float(comp_raw.get(k) or 0.0))
            worst_val = float(comp_raw.get(worst_key) or 0.0)
            metric = {
                "component": comp_name,
                "metric": worst_key,
                "value": worst_val,
                "note": f"Lowest subscore in {comp_name}: {worst_key} = {worst_val:.3f}",
            }
        else:
            numeric = {k: v for k, v in comp_raw.items() if isinstance(v, (int, float))}
            if numeric:
                vals = list(numeric.values())
                med = pd.Series(vals).median()
                far_key = max(numeric.keys(), key=lambda k: abs(numeric[k] - med))
                metric = {
                    "component": comp_name,
                    "metric": far_key,
                    "value": numeric[far_key],
                    "note": f"Strong signal in {comp_name}: {far_key} = {numeric[far_key]}",
                }
            else:
                metric = {"component": comp_name, "metric": None, "value": None, "note": f"No numeric details for {comp_name}"}
        selected.append(metric)
    return selected


@app.post("/score", response_class=JSONResponse)
async def score_file(file: UploadFile = File(...)):
    try:
        df = _read_uploaded_file(file)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Normalize common column names
    if "amount" not in df.columns and "amt" in df.columns:
        df = df.rename(columns={"amt": "amount"})
    if "description" not in df.columns and "memo" in df.columns:
        df = df.rename(columns={"memo": "description"})

    # Categorize: prefer ML model if present and predict wrapper available
    df_cat = None
    if ML_MODEL is not None and PREDICT_AVAILABLE:
        try:
            df_cat = predict_categories(df, text_cols=("description", "payee"), model=ML_MODEL, out_col="category")
            logger.info("Categorized transactions using ML model (startup-loaded).")
        except Exception as e:
            logger.exception("ML categorizer failed, falling back to rule-based: %s", e)
            df_cat = None

    if df_cat is None:
        try:
            df_cat = categorize_transactions(df, text_cols=("description", "payee"))
            logger.info("Categorized transactions using rule-based classifier.")
        except Exception:
            try:
                df_cat = df.copy()
                df_cat["category"] = "Uncategorized"
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to categorize transactions: {e}")

    # Compute features
    try:
        features = compute_all_features(df_cat)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    # Score & eligibility & risk & explanations
    try:
        scoring = compute_final_score(features["component_scores"])
        eligibility = evaluate_loan_eligibility(features, scoring["final_score"])
        risk = risk_bucket_from_score(scoring["final_score"])
        explanations = generate_explanations(features["raw"])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {e}")

    try:
        top_features = _pick_top_features(features["component_scores"], features["raw"], top_k=2)
    except Exception:
        top_features = []

    response = {
        "component_scores": features["component_scores"],
        "final_score": scoring,
        "eligibility": eligibility,
        "risk": risk,
        "explanations": explanations,
        "top_features": top_features,
    }
    return JSONResponse(content=response)
