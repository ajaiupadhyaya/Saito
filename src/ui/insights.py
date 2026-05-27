"""Narrative insight helpers for portfolio-ready UX."""

from __future__ import annotations


def strategy_narrative(ann_return: float, sharpe: float, max_drawdown: float, overfit_gap: float) -> str:
    tone = "stable" if max_drawdown > -0.15 else "fragile"
    edge = "strong" if sharpe >= 1.0 else "moderate" if sharpe >= 0.5 else "weak"
    overfit = "low" if overfit_gap <= 0.2 else "elevated" if overfit_gap <= 0.5 else "high"
    return (
        f"Signal edge appears {edge} with {tone} drawdown behavior. "
        f"Annualized return is {ann_return:.2%}, while walk-forward overfit risk is {overfit} "
        f"(train-test Sharpe gap {overfit_gap:.2f})."
    )

