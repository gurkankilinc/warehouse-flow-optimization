"""Clean the raw retail transactions into a tidy outbound-demand table.

The raw file is genuinely messy, which is the point: it exercises the same
cleaning judgement any real project needs. The issues handled here are

  * two different column namings across dataset versions
    (``InvoiceNo``/``UnitPrice``/``CustomerID`` vs ``Invoice``/``Price``/``Customer ID``)
  * cancellation invoices, flagged by a ``C`` prefix, carrying negative quantities
  * stock codes that are not products at all (postage, bank charges, manual
    adjustments, test rows)
  * absurd quantities and prices from data-entry errors
  * exact duplicate rows

Every step records how many rows it removed, so the cleaning is auditable
instead of being a black box -- ``clean_transactions`` returns that audit trail
alongside the cleaned frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CONFIG, CalibrationConfig

__all__ = ["load_raw", "clean_transactions"]

# Maps every known raw spelling onto the canonical name used downstream.
_COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_datetime",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "customer id": "customer_id",
    "country": "country",
}

_REQUIRED = ("invoice", "stock_code", "quantity", "invoice_datetime", "unit_price")


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", "")
        if key in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[key]
        elif str(col).strip().lower() in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[str(col).strip().lower()]
    df = df.rename(columns=renamed)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"raw data is missing required columns: {missing}")
    return df


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw dataset, concatenating all sheets if it is an Excel file."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        frames = [_canonicalise_columns(s) for s in sheets.values()]
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = _canonicalise_columns(pd.read_csv(path, encoding="latin-1"))
    return raw


def clean_transactions(
    raw: pd.DataFrame, cfg: CalibrationConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning rules.

    Returns
    -------
    clean : tidy transaction lines representing genuine outbound demand
    audit : one row per cleaning step with the row count it removed
    """
    cfg = cfg or CONFIG.calibration
    df = raw.copy()
    audit: list[dict[str, object]] = []

    def record(step: str, before: int) -> None:
        audit.append({"step": step, "rows_removed": before - len(df), "rows_remaining": len(df)})

    audit.append({"step": "raw", "rows_removed": 0, "rows_remaining": len(df)})

    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()

    n = len(df)
    df = df.dropna(subset=["invoice", "stock_code", "quantity", "unit_price", "invoice_datetime"])
    record("drop rows with missing key fields", n)

    n = len(df)
    df = df[~df["invoice"].str.startswith(cfg.cancelled_invoice_prefix)]
    record("drop cancellation invoices", n)

    n = len(df)
    non_product = {code.strip().upper() for code in cfg.non_product_codes}
    df = df[~df["stock_code"].isin(non_product)]
    record("drop non-product stock codes", n)

    # Real product codes are 5 digits, optionally with a letter variant suffix.
    n = len(df)
    df = df[df["stock_code"].str.match(r"^\d{5}[A-Z]*$", na=False)]
    record("drop malformed stock codes", n)

    n = len(df)
    df = df[df["quantity"].between(cfg.min_quantity, cfg.max_quantity)]
    record("drop out-of-range quantities (incl. returns)", n)

    n = len(df)
    df = df[df["unit_price"].between(cfg.min_unit_price, cfg.max_unit_price)]
    record("drop out-of-range prices", n)

    n = len(df)
    df = df.drop_duplicates()
    record("drop exact duplicate rows", n)

    df["invoice_datetime"] = pd.to_datetime(df["invoice_datetime"], errors="coerce")
    n = len(df)
    df = df.dropna(subset=["invoice_datetime"])
    record("drop unparseable timestamps", n)

    df["invoice_date"] = df["invoice_datetime"].dt.normalize()
    df["line_revenue"] = df["quantity"] * df["unit_price"]

    keep = [
        "invoice", "stock_code", "description", "quantity", "unit_price",
        "line_revenue", "invoice_datetime", "invoice_date", "customer_id", "country",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    return df, pd.DataFrame(audit)
