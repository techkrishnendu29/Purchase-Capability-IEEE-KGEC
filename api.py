#!/usr/bin/env python3
# scripts/api.py
from __future__ import annotations
import os
import tempfile
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Pipeline imports (must be importable from repo root)
from preprocessing.loader import load_transaction_files
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

# Optional ML helpers (may not exist in all setups)
try:
    from preprocessing.ml_categorizer import load_model as ml_load_model, predict_categories  # type: ignore
    ML_HELPERS_AVAILABLE = True
except Exception:
    ml_load_model = None  # type: ignore
    predict_categories = None  # type: ignore
    ML_HELPERS_AVAILABLE = False

# fallback joblib load if needed
try:
    from joblib import load as joblib_load  # type: ignore
except Exception:  # joblib may not be installed; it's recommended in requirements.txt
    joblib_load = None  # type: ignore

logger = logging.getLogger("prosperity_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="ProsperityScore API (final)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Model config from environment
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/ml_model.pkl"))
USE_ML_ENV = os.environ.get("USE_ML", "false").lower()
REQUEST_USE_ML = USE_ML_ENV in ("1", "true", "yes")

# In-memory ML model object (if loaded)
ML_MODEL = None
ML_MODEL_LOADED = False

def try_load_ml_model():
    global ML_MODEL, ML_MODEL_LOADED
    if not REQUEST_USE_ML:
        logger.info("USE_ML not enabled; ML categorizer disabled.")
        ML_MODEL = None
        ML_MODEL_LOADED = False
        return

    if not MODEL_PATH.exists():
        logger.warning("USE_ML enabled but MODEL_PATH not found: %s", MODEL_PATH)
        ML_MODEL = None
        ML_MODEL_LOADED = False
        return

    # Preferred: use preprocessing.ml_categorizer.load_model if available
    if ML_HELPERS_AVAILABLE and ml_load_model is not None:
        try:
            ML_MODEL = ml_load_model(str(MODEL_PATH))
            ML_MODEL_LOADED = True
            logger.info("Loaded ML model via ml_categorizer.load_model from %s", MODEL_PATH)
            return
        except Exception as e:
            logger.exception("ml_categorizer.load_model failed: %s", e)

    # Fallback: joblib.load if available
    if joblib_load is not None:
        try:
            ML_MODEL = joblib_load(str(MODEL_PATH))
            ML_MODEL_LOADED = True
            logger.info("Loaded ML model via joblib from %s", MODEL_PATH)
            return
        except Exception as e:
            logger.exception("joblib.load failed to load model at %s: %s", MODEL_PATH, e)

    logger.warning("ML model could not be loaded even though USE_ML=true and MODEL_PATH exists.")
    ML_MODEL = None
    ML_MODEL_LOADED = False

# load ML model at startup attempt
try_load_ml_model()


@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse(
        content={
            "status": "ok",
            "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
            "use_ml_requested": REQUEST_USE_ML,
            "ml_helpers_available": ML_HELPERS_AVAILABLE,
            "ml_model_loaded": ML_MODEL_LOADED,
            "cwd": str(Path.cwd()),
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Simple landing + upload form to test /score."""
    health_json = health().body.decode() if isinstance(health().body, (bytes, bytearray)) else str(health().body)
    html = f"""
    <!doctype html>
    <html>
      <head><meta charset="utf-8"/><title>ProsperityScore API</title></head>
      <body style="font-family:Arial,Helvetica,sans-serif; margin:30px;">
        <h1>ProsperityScore API</h1>
        <p>Health: <pre>{health_json}</pre></p>
        <h3>Upload a CSV / XLS / XLSX bank statement</h3>
        <input id="file" type="file" accept=".csv,.xls,.xlsx"/>
        <button id="send">Upload & Score</button>
        <pre id="out"></pre>
        <script>
          document.getElementById('send').onclick = async () => {{
            const f = document.getElementById('file').files[0];
            if (!f) return alert('Select a file');
            const fd = new FormData();
            fd.append('file', f);
            document.getElementById('out').textContent = 'Uploading...';
            try {{
              const resp = await fetch('{request.url_for("score_file")}', {{ method: 'POST', body: fd }});
              const j = await resp.json();
              document.getElementById('out').textContent = JSON.stringify(j, null, 2);
            }} catch (e) {{
              document.getElementById('out').textContent = 'Error: ' + e;
            }}
          }};
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


def _pick_top_features(component_scores: Dict[str, float], raw: Dict[str, Any], top_k: int = 2) -> List[Dict[str, Any]]:
    comps_sorted = sorted(component_scores.items(), key=lambda kv: abs(kv[1] - 50), reverse=True)
    selected = []
    for comp_name, _ in comps_sorted[:top_k]:
        comp_raw = raw.get(f"{comp_name}_raw") or raw.get(comp_name) or {}
        if not isinstance(comp_raw, dict):
            comp_raw = getattr(comp_raw, "__dict__", {}) or {}
        score_keys = [k for k in comp_raw.keys() if k.endswith("_score")]
        if score_keys:
            worst_key = min(score_keys, key=lambda k: float(comp_raw.get(k) or 0.0))
            selected.append({
                "component": comp_name,
                "metric": worst_key,
                "value": float(comp_raw.get(worst_key) or 0.0),
                "note": f"Lowest subscore: {worst_key}"
            })
        else:
            numeric = {k: v for k, v in comp_raw.items() if isinstance(v, (int, float))}
            if numeric:
                key = max(numeric.keys(), key=lambda kk: abs(numeric[kk]))
                selected.append({
                    "component": comp_name,
                    "metric": key,
                    "value": numeric[key],
                    "note": f"Strong numeric signal: {key}"
                })
            else:
                selected.append({"component": comp_name, "metric": None, "value": None, "note": "No numeric detail"})
    return selected


@app.post("/score", response_class=JSONResponse)
async def score_file(file: UploadFile = File(...)):
    """
    Accept one uploaded CSV/XLS/XLSX file, parse using preprocessing.loader.load_transaction_files,
    categorize (ML if enabled and available, else rule-based), compute features, score & return JSON.
    """
    fname = (file.filename or "upload").strip()
    ext = Path(fname).suffix.lower()
    if ext not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .xls, or .xlsx")

    tmp_path = None
    try:
        # write uploaded content to a temporary file for loader to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
            tmp.flush()

        # parse & normalize using your loader
        try:
            df = load_transaction_files([tmp_path])
        except Exception as e:
            logger.exception("Loader failed to parse uploaded file: %s", e)
            # return helpful 400
            raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file parsed but returned no transactions")

        # Categorize: prefer ML if enabled and available
        df_cat = None
        if REQUEST_USE_ML and ML_MODEL_LOADED and predict_categories is not None:
            try:
                # predict_categories signature in your code accepts model and out_col
                df_cat = predict_categories(df, text_cols=("description", "payee"), model=ML_MODEL, out_col="category")
                logger.info("Categorized transactions using ML predict_categories")
            except Exception as e:
                logger.exception("ML predict_categories failed, falling back to rule-based: %s", e)
                df_cat = None

        if df_cat is None:
            try:
                # categorize_transactions from preprocessing.categorizer will use external rule module if present
                df_cat = categorize_transactions(df, text_cols=("description", "payee"), use_ml=False)
                logger.info("Categorized transactions using rule-based categorize_transactions")
            except Exception as e:
                logger.exception("Rule-based categorizer failed; proceeding with uncategorized rows: %s", e)
                df_cat = df.copy()
                df_cat["category"] = "Uncategorized"

        # Compute features
        try:
            features = compute_all_features(df_cat)
        except Exception as e:
            logger.exception("Feature computation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

        # Scoring, eligibility, risk, explanations
        try:
            scoring = compute_final_score(features["component_scores"])
            eligibility = evaluate_loan_eligibility(features, scoring["final_score"])
            risk = risk_bucket_from_score(scoring["final_score"])
            explanations = generate_explanations(features["raw"])
        except Exception as e:
            logger.exception("Scoring pipeline failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {e}")

        # Top features heuristic
        try:
            top_features = _pick_top_features(features["component_scores"], features["raw"], top_k=2)
        except Exception:
            top_features = []

        resp = {
            "component_scores": features["component_scores"],
            "final_score": scoring,
            "eligibility": eligibility,
            "risk": risk,
            "explanations": explanations,
            "top_features": top_features,
        }
        return JSONResponse(content=resp)

    finally:
        # cleanup temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.exception("Failed to remove temporary file %s", tmp_path)
