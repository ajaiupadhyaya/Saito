"""Shared Plotly styling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def apply_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_to_bytes(fig: go.Figure, fmt: str = "html") -> bytes:
    if fmt == "html":
        return fig.to_html(include_plotlyjs="cdn").encode()
    return fig.to_image(format=fmt)


def correlation_heatmap(corr: pd.DataFrame | np.ndarray, title: str) -> go.Figure:
    arr = corr.values if hasattr(corr, "values") else np.asarray(corr, dtype=float)
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale="RdBu", zmin=-1, zmax=1))
    return apply_theme(fig, title)


def eigenvalue_bar(values: np.ndarray, title: str) -> go.Figure:
    vals = np.asarray(values, dtype=float).ravel()
    fig = go.Figure(go.Bar(x=list(range(1, len(vals) + 1)), y=vals))
    return apply_theme(fig, title)
