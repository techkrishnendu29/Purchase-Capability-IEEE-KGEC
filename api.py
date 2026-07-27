#!/usr/bin/env python3
# scripts/api.py
from __future__ import annotations
import logging
import traceback
from io import BytesIO
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

# Pipeline imports (must be importable from repo root)
from preprocessing.categorizer import categorize_transactions
from feature_engineering.feature_engineering import compute_all_features
from scoring.credit_score import compute_final_score
from scoring.loan_eligibility import evaluate_loan_eligibility
from scoring.risk_analysis import risk_bucket_from_score
from scoring.explainability import generate_explanations

logger = logging.getLogger("prosperity_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="ProsperityScore API (minimal)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Try to load a local ML model if present (optional). If missing, we simply use rule-based.
MODEL_PATH = Path("models/ml_model.pkl")
ML_MODEL = None
try:
    if MODEL_PATH.exists():
        ML_MODEL = load(str(MODEL_PATH))
        logger.info("Loaded ML model from %s", MODEL_PATH)
    else:
        logger.info("No ML model found at %s; using rule-based categorizer", MODEL_PATH)
except Exception:
    logger.exception("Failed to load ML model; continuing with rule-based categorizer")
    ML_MODEL = None


@app.get("/health", response_class=JSONResponse)
def health():
    """Simple health endpoint."""
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": ML_MODEL is not None,
            "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
            "cwd": str(Path.cwd()),
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Landing page with a small upload form to test /score."""
    health_json = health().body.decode() if isinstance(health().body, (bytes, bytearray)) else str(health().body)
    html = f"""
    <!doctype html>
    <html>
      <head><meta charset="utf-8"/><title>ProsperityScore API</title></head>
      <body style="font-family:Arial,Helvetica,sans-serif; margin:30px;">
        <h1>ProsperityScore API</h1>
        <p>Health: <pre>{health_json}</pre></p>
        <h3>Test upload</h3>
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


def _read_uploaded_file(upload_file: UploadFile) -> pd.DataFrame:
    """Read uploaded CSV/XLS(X) into a pandas DataFrame with encoding fallback for CSVs."""
    content = upload_file.file.read()
    upload_file.file.close()
    bio = BytesIO(content)
    fname = (upload_file.filename or "").lower()
    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            return pd.read_excel(bio)
        if fname.endswith(".csv"):
            try:
                bio.seek(0)
                return pd.read_csv(bio)
            except Exception:
                bio.seek(0)
                return pd.read_csv(bio, encoding="latin1")
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .xls, or .xlsx")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to parse uploaded file: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")


def _pick_top_features(component_scores: Dict[str, float], raw: Dict[str, Any], top_k: int = 2) -> List[Dict[str, Any]]:
    """Simple heuristic to pick 1-2 most valuable features per component scores and raw fields."""
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
    """Accept one CSV/XLS(X) transactions file and return scoring JSON."""
    try:
        df = _read_uploaded_file(file)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # normalize common column names
    if "amount" not in df.columns and "amt" in df.columns:
        df = df.rename(columns={"amt": "amount"})
    if "description" not in df.columns and "memo" in df.columns:
        df = df.rename(columns={"memo": "description"})

    # categorize (rule-based; ML model not used here to keep minimal & robust)
    try:
        df_cat = categorize_transactions(df, text_cols=("description", "payee"))
    except Exception:
        # fallback: don't fail; continue with uncategorized data
        logger.exception("Rule-based categorizer failed; falling back to uncategorized")
        df_cat = df.copy()
        df_cat["category"] = "Uncategorized"

    # compute features
    try:
        features = compute_all_features(df_cat)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    # scoring
    try:
        scoring = compute_final_score(features["component_scores"])
        eligibility = evaluate_loan_eligibility(features, scoring["final_score"])
        risk = risk_bucket_from_score(scoring["final_score"])
        explanations = generate_explanations(features["raw"])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {e}")

    # pick top features
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
