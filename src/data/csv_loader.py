"""Parse user-uploaded CSV (rows=time, cols=assets)."""

from __future__ import annotations

import io

import pandas as pd


def load_csv_bytes(data: bytes, index_col: int | None = 0) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), index_col=index_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().all():
        df = df.reset_index(drop=True)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("CSV must contain numeric columns for asset prices or returns")
    numeric = numeric.dropna(how="all")
    numeric.columns = [str(c) for c in numeric.columns]
    return numeric.sort_index()
