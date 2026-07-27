# scripts/api.py
from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from io import BytesIO
from pathlib import Path
import pandas as pd
import traceback

# Ensure repo root on sys.path if needed (uncomment if running from other cwd)
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import your existing pipeline functions
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

app = FastAPI(title="ProsperityScore API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _read_uploaded_file(upload_file: UploadFile) -> pd.DataFrame:
    content = upload_file.file.read()
    upload_file.file.close()
    bio = BytesIO(content)
    fname = upload_file.filename.lower()
    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(bio)
        elif fname.endswith(".csv"):
            # try utf-8 then fallback
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
    """
    Heuristic:
      - pick top components by absolute component score contribution,
      - within each chosen component, look for raw "*_score" fields and choose the lowest (weakest) or highest (strongest)
        depending on whether it indicates risk or strength. Return label + metric name + value + human reason.
    This is a heuristic for "1-2 most valuable features" to present in the API.
    """
    # sort components by importance (difference from neutral 50)
    comps_sorted = sorted(component_scores.items(), key=lambda kv: abs(kv[1] - 50), reverse=True)
    selected = []
    for comp_name, comp_val in comps_sorted[:top_k]:
        comp_raw = raw.get(f"{comp_name}_raw") or raw.get(comp_name) or {}
        # convert dataclass/object to dict if needed
        if not isinstance(comp_raw, dict):
            try:
                comp_raw = getattr(comp_raw, "__dict__", dict(comp_raw))
            except Exception:
                comp_raw = dict()
        # find *_score keys
        score_keys = [k for k in comp_raw.keys() if k.endswith("_score")]
        metric = None
        reason = None
        if score_keys:
            # for risk-oriented presentation, pick worst score for that component (lowest)
            worst_key = min(score_keys, key=lambda k: float(comp_raw.get(k) or 0.0))
            worst_val = float(comp_raw.get(worst_key) or 0.0)
            metric = {"component": comp_name, "metric": worst_key, "value": worst_val,
                      "note": f"Lowest subscore in {comp_name}: {worst_key} = {worst_val:.3f}"}
        else:
            # fallback: pick numeric field with largest absolute deviation from median
            numeric = {k: v for k, v in comp_raw.items() if isinstance(v, (int, float))}
            if numeric:
                # choose metric farthest from median of that component
                vals = list(numeric.values())
                med = pd.Series(vals).median()
                far_key = max(numeric.keys(), key=lambda k: abs(numeric[k] - med))
                metric = {"component": comp_name, "metric": far_key, "value": numeric[far_key],
                          "note": f"Strong signal in {comp_name}: {far_key} = {numeric[far_key]}"}
            else:
                metric = {"component": comp_name, "metric": None, "value": None, "note": f"No numeric details for {comp_name}"}
        selected.append(metric)
    return selected

@app.post("/score", response_class=JSONResponse)
async def score_file(file: UploadFile = File(...)):
    """
    Upload one CSV/XLS(X) file containing transactions (columns like date, amount, description, payee).
    Returns a JSON with component scores, final score, credit score, eligibility, risk bucket,
    explanations and the 1-2 most valuable features.
    """
    try:
        df = _read_uploaded_file(file)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Basic normalization: ensure date, description, payee, amount columns exist depending on loader
    # If your loader provides custom normalization, swap in that call.
    # Here we expect at least 'amount' and 'description' or 'payee' present.
    if "amount" not in df.columns and "amt" in df.columns:
        df = df.rename(columns={"amt": "amount"})
    if "description" not in df.columns and "memo" in df.columns:
        df = df.rename(columns={"memo": "description"})

    # Categorize transactions (rule-based)
    try:
        df_cat = categorize_transactions(df, text_cols=("description", "payee"))
    except Exception:
        # fallback: if categorize_transactions expects different columns, attempt minimal fallback
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

    # pick top 1-2 features
    try:
        top_features = _pick_top_features(features["component_scores"], features["raw"], top_k=2)
    except Exception:
        top_features = []

    # format response
    response = {
        "component_scores": features["component_scores"],
        "final_score": scoring,
        "eligibility": eligibility,
        "risk": risk,
        "explanations": explanations,
        "top_features": top_features,
    }
    return JSONResponse(content=response)