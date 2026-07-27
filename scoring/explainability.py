"""
Produce prioritized, human-readable reasons (3-4) explaining score & decision.
The function builds candidate reasons with impact scores, ranks them, and returns
the top max_reasons messages (prefer negative/important signals first, then strengths).
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


def _get(f: Optional[object], key: str, default=None):
    """Support both dicts and objects with attributes."""
    if f is None:
        return default
    if isinstance(f, dict):
        return f.get(key, default)
    return getattr(f, key, default)


def generate_explanations(features: Dict[str, Any], max_reasons: int = 4) -> List[str]:
    candidates: List[Tuple[float, str]] = []

    income = features.get("income", {}) or {}
    expense = features.get("expense", {}) or {}
    cashflow = features.get("cashflow", {}) or {}
    repayment = features.get("repayment", {}) or {}
    behaviour = features.get("behaviour", {}) or {}

    # Helpful numeric shortcuts
    inc_comp = _get(income, "income_component_score")
    inc_growth = _get(income, "income_growth_score") or 0.0
    avg_income = _get(income, "avg_monthly_income")

    exp_comp = _get(expense, "expense_component_score")
    eir = _get(expense, "eir")  # expense-to-income metric if present
    disc_ratio = _get(expense, "discretionary_ratio") or 0.0
    total_expenses = _get(expense, "total_expenses")

    cash_comp = _get(cashflow, "cashflow_component_score")
    avg_balance = _get(cashflow, "avg_monthly_balance")
    neg_bal_count = _get(cashflow, "negative_balance_count") or 0
    min_balance = _get(cashflow, "min_monthly_balance")

    emi_count = _get(repayment, "emi_count") or 0
    on_time_ratio = _get(repayment, "on_time_ratio")
    bounce_count = _get(repayment, "bounce_count") or 0

    savings_rate = _get(behaviour, "savings_rate")
    discipline = _get(behaviour, "financial_discipline_score")
    spike = _get(behaviour, "spending_spike_score")
    eom_stress = _get(behaviour, "end_of_month_stress_score")

    # 1) Income-related reasons
    if inc_comp is not None:
        if inc_comp < 40:
            candidates.append((9.0, f"Income weak or unstable (income score {inc_comp:.0f}/100)."))
        elif inc_comp < 70:
            candidates.append((5.0, f"Moderate income stability (income score {inc_comp:.0f}/100)."))
        else:
            candidates.append((2.0, f"Income strength: {inc_comp:.0f}/100."))

    # Income growth trend
    if inc_growth is not None:
        if inc_growth < 0.4:
            # negative or low growth is more urgent when average income is also falling/low
            severity = 7.0 if (avg_income is None or (isinstance(avg_income, (int, float)) and avg_income < 50000)) else 5.0
            candidates.append((severity, f"Income growth weak (growth score {inc_growth:.2f}); recent decline or stagnation detected."))
        else:
            candidates.append((1.0, f"Income growth looks steady (growth score {inc_growth:.2f})."))

    # 2) Expense-related reasons
    if eir is not None:
        # thresholds can be tuned; higher eir -> more risky
        if eir >= 0.6:
            candidates.append((9.5, f"Very high expense-to-income ratio ({eir:.2f}); limited spare capacity."))
        elif eir >= 0.4:
            candidates.append((6.0, f"Elevated expense-to-income ratio ({eir:.2f}); consider reducing discretionary spend."))
    if total_expenses is not None and avg_income:
        # highlight if expenses occupy a large share relative to income (fallback)
        try:
            if avg_income > 0 and (total_expenses / avg_income) >= 0.6:
                candidates.append((6.5, "Expenses are a large share of income; lowering expenses would improve affordability."))
        except Exception:
            pass
    if disc_ratio and disc_ratio > 0.4:
        candidates.append((5.5, f"High discretionary spending (ratio {disc_ratio:.2f}) — potential to cut non-essential outflows."))

    # 3) Cashflow and liquidity
    if neg_bal_count and neg_bal_count > 0:
        candidates.append((9.0, f"Negative balance occurrences detected ({neg_bal_count} times) — indicates end-of-month stress."))
    elif min_balance is not None and avg_balance is not None:
        # low min vs avg might indicate volatility
        if isinstance(min_balance, (int, float)) and isinstance(avg_balance, (int, float)):
            if min_balance < 0.2 * avg_balance:
                candidates.append((6.5, "Minimum monthly balance is much lower than average, indicating volatility in liquidity."))

    if cash_comp is not None and cash_comp < 50:
        candidates.append((6.0, f"Cashflow stability is low (cashflow score {cash_comp:.0f}/100)."))

    # 4) Repayment behaviour
    if bounce_count and bounce_count > 0:
        candidates.append((10.0, f"Transaction bounces detected ({bounce_count}) — strong negative signal for repayment reliability."))
    elif on_time_ratio is not None:
        if on_time_ratio < 0.9:
            candidates.append((8.5, f"Repayment timeliness below ideal (on-time ratio {on_time_ratio:.2f}) — risk of missed payments."))
        else:
            candidates.append((2.0, f"Repayment timeliness good (on-time ratio {on_time_ratio:.2f})."))
    # many EMIs relative to income could be important; we only flag when emi_count is large
    if emi_count and emi_count >= 4:
        candidates.append((7.0, f"Multiple active EMIs ({emi_count}) may constrain new borrowing capacity."))

    # 5) Behavioural signals
    if savings_rate is not None:
        if savings_rate < 0.1:
            candidates.append((8.0, f"Low savings rate ({savings_rate:.2f}) — weak buffer to absorb shocks."))
        elif savings_rate < 0.4:
            candidates.append((4.0, f"Moderate savings rate ({savings_rate:.2f})."))
        else:
            candidates.append((1.5, f"Healthy savings rate ({savings_rate:.2f})."))

    if discipline is not None and discipline < 0.4:
        candidates.append((7.5, "Low financial discipline score — irregular savings or spending behavior detected."))
    if spike is not None and spike > 0.7:
        candidates.append((6.0, "Recent large spending spikes detected — potential volatility in budget."))

    if eom_stress is not None and eom_stress < 0.4:
        candidates.append((6.5, "End-of-month stress indicator low — months may end with tight cash positions."))

    # If we have no strong negative signals, add top strengths to explain approval
    # strengths get lower priority scores than negatives
    strengths = []
    # repayment perfect
    rep_comp = _get(repayment, "repayment_component_score")
    if rep_comp is not None and rep_comp >= 95:
        strengths.append((2.5, "Excellent repayment history (no bounces / on-time payments)."))
    if exp_comp is not None and exp_comp >= 85:
        strengths.append((2.0, f"Expenses well-managed (expense score {exp_comp:.0f}/100)."))
    if cash_comp is not None and cash_comp >= 80:
        strengths.append((2.0, f"Strong cashflow stability (cashflow score {cash_comp:.0f}/100)."))
    if discipline is not None and discipline >= 0.8:
        strengths.append((1.5, "High financial discipline (consistent savings & budgeting)."))

    # Combine candidates: prioritize negative/high-severity, but keep some strengths if few negatives
    # Sort candidates by score descending
    candidates_sorted = sorted(candidates, key=lambda x: -x[0])

    selected: List[str] = []
    # pick top negative/important signals first
    for score, msg in candidates_sorted:
        if len(selected) >= max_reasons:
            break
        # treat messages with high score (>=5) as important signals
        if score >= 5:
            selected.append(msg)

    # if not enough, add mid/low severity signals
    if len(selected) < max_reasons:
        for score, msg in candidates_sorted:
            if len(selected) >= max_reasons:
                break
            if msg not in selected:
                selected.append(msg)

    # if still fewer than max_reasons, append strengths to reach desired count
    if len(selected) < max_reasons:
        for score, msg in sorted(strengths, key=lambda x: -x[0]):
            if len(selected) >= max_reasons:
                break
            if msg not in selected:
                selected.append(msg)

    # final fallback
    if not selected:
        selected = ["No glaring risk signals detected"]

    # Clean up messages (truncate long numbers) and return
    def _fmt(m: str) -> str:
        return m.replace("\n", " ").strip()

    return [_fmt(s) for s in selected[:max_reasons]]