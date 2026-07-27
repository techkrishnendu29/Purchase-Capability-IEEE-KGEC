"""
Income features module.

API:
- compute_income_features(df, date_col='date', amount_col='amount', source_col='payee')
  -> IncomeFeatures dataclass

Expected input:
- df: pandas.DataFrame including at minimum a date column and an amount column.
  Income transactions are identified as positive amounts by default. If your categorizer tags 'category' you can filter externally.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class IncomeFeatures:
    avg_monthly_income: float
    income_stability_score: float        # 0..1
    income_consistency_score: float     # 0..1 (alias of stability by default)
    income_source_diversity_score: float # 0..1 (higher = safer per your spec)
    income_growth_slope: float          # raw slope (currency per month)
    income_growth_score: float          # 0..1 normalized
    income_component_score: float       # 0..100 final normalized income score for pipeline use


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.Series:
    s = df[date_col]
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, errors="coerce")
    return s


def compute_monthly_income_series(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
) -> pd.Series:
    """
    Return a pandas Series indexed by month period (YYYY-MM) with aggregated income (sum of positive amounts).
    """
    if date_col not in df.columns or amount_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns {date_col!r} and {amount_col!r}")

    dates = _ensure_datetime(df, date_col)
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    temp = pd.DataFrame({ "date": dates, "amount": amounts })
    temp = temp.dropna(subset=["date"])
    # consider incomes as positive amounts
    incomes = temp[temp["amount"] > 0].copy()
    if incomes.empty:
        # return empty series with monthly period index
        return pd.Series(dtype=float)

    incomes["period"] = incomes["date"].dt.to_period("M")
    monthly = incomes.groupby("period")["amount"].sum().sort_index()
    # convert PeriodIndex to datetime period start for regressions if needed
    monthly.index = monthly.index.to_timestamp()
    return monthly


def _normalize_0_1(x: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    v = (x - min_val) / (max_val - min_val)
    return float(np.clip(v, 0.0, 1.0))


def compute_income_features(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    source_col: Optional[str] = "payee",
    min_months_for_growth: int = 3,
) -> IncomeFeatures:
    """
    Compute income-related features from transactions DataFrame.

    Returns an IncomeFeatures dataclass where:
    - stability/consistency: 1 - (std / mean) clipped to 0..1
    - source diversity score: normalized inverse of number of unique payers/sources (1 = single consistent source)
    - growth_score: based on linear regression slope; normalized against historical absolute values
    - income_component_score: combined metric mapped to 0..100 for the final pipeline
    """

    monthly = compute_monthly_income_series(df, date_col, amount_col)

    if monthly.empty:
        logger.debug("No positive income transactions found.")
        return IncomeFeatures(
            avg_monthly_income=0.0,
            income_stability_score=0.0,
            income_consistency_score=0.0,
            income_source_diversity_score=0.0,
            income_growth_slope=0.0,
            income_growth_score=0.0,
            income_component_score=0.0,
        )

    avg_monthly = float(monthly.mean())
    std_monthly = float(monthly.std(ddof=0))

    # Income stability/consistency: 1 - (std/mean) clipped to 0..1
    if avg_monthly > 0:
        stability = 1.0 - (std_monthly / avg_monthly)
    else:
        stability = 0.0
    stability = float(np.clip(stability, 0.0, 1.0))

    # Source diversity: fewer unique sources = higher safety per your spec.
    if source_col and source_col in df.columns:
        sources = df.loc[pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) > 0, source_col]
        unique_sources = sources.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
        n_sources = max(1, len(unique_sources))
    else:
        # fallback: use description if payee not available
        if "description" in df.columns:
            sources = df.loc[pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0) > 0, "description"]
            unique_sources = sources.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
            n_sources = max(1, len(unique_sources))
        else:
            n_sources = 1

    # Convert number of sources to a score where 1.0 = single source, and decreases with more sources.
    # We'll use 1 / log(1 + n_sources) normalized to 0..1 for better scaling. Then clip.
    diversity_raw = 1.0 / np.log1p(n_sources)
    # normalize diversity_raw to 0..1 given reasonable bounds: [1/log(1+1), 1/log(1+10)] -> map to [1, ~0.43]; we clamp
    diversity_score = float(np.clip((diversity_raw - 1.0) / (1.0 - (1.0/np.log1p(10))), -1.0, 1.0))
    # simpler transform: map single source to 1.0, many sources (>=6) -> 0.0
    if n_sources <= 1:
        diversity_score = 1.0
    elif n_sources >= 6:
        diversity_score = 0.0
    else:
        diversity_score = float(np.clip(1.0 - (n_sources - 1) / 5.0, 0.0, 1.0))

    # Income growth: linear regression on monthly totals (months as integer)
    income_growth_slope = 0.0
    growth_score = 0.0
    if len(monthly) >= min_months_for_growth:
        X = np.arange(len(monthly)).reshape(-1, 1)  # month index
        y = monthly.values.reshape(-1, 1)
        reg = LinearRegression()
        try:
            reg.fit(X, y)
            slope = float(reg.coef_[0][0])  # currency per month
            income_growth_slope = slope

            # Normalize slope to a score in 0..1.
            # We consider a reasonable slope range: [-mean, +mean], where mean = avg_monthly.
            # Positive slope gives higher score; negative slope lowers it.
            if avg_monthly > 0:
                growth_score = float(np.clip((slope / avg_monthly) * 0.5 + 0.5, 0.0, 1.0))
            else:
                growth_score = 0.5 if slope == 0 else (1.0 if slope > 0 else 0.0)
        except Exception as e:
            logger.exception("Income growth regression failed: %s", e)
            income_growth_slope = 0.0
            growth_score = 0.0

    # Compose a final income component score 0..100:
    # weights (example): avg_monthly normalized via log, stability 40%, diversity 20%, growth 40%
    # Normalize average monthly income via log1p against itself + some smoothing to 0..1
    avg_norm = 0.0
    try:
        # use log scale to reduce sensitivity to outliers
        avg_norm = float(np.clip(np.log1p(avg_monthly) / (np.log1p(avg_monthly) + 1.0), 0.0, 1.0))
    except Exception:
        avg_norm = 0.0

    # final combined in 0..1
    combined = 0.0
    # weights chosen to reflect Income weight ~30% in the final score; internal composition can differ
    w_avg = 0.3
    w_stab = 0.4
    w_div = 0.15
    w_growth = 0.15

    combined = w_avg * avg_norm + w_stab * stability + w_div * diversity_score + w_growth * growth_score
    income_component_score = float(np.clip(combined * 100.0, 0.0, 100.0))

    return IncomeFeatures(
        avg_monthly_income=avg_monthly,
        income_stability_score=stability,
        income_consistency_score=stability,
        income_source_diversity_score=diversity_score,
        income_growth_slope=income_growth_slope,
        income_growth_score=growth_score,
        income_component_score=income_component_score,
    )