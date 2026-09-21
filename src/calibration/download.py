"""Fetch the real retail transaction dataset used to calibrate demand.

Why a real dataset at all? The warehouse *geometry* has to be synthetic -- no
public dataset contains real slot layouts. But the *demand* side does not have
to be invented: popularity skew, basket size and seasonality can be estimated
from real transactions. That is the difference between "I made up some numbers"
and "I estimated the parameters from data", and it is the single cheapest way
to make a simulation-based project credible.

If the download is unavailable (offline, or the host moved the file) the
pipeline does not break: ``fit_parameters`` falls back to documented parametric
defaults and records ``source = "fallback"`` so every downstream report stays
honest about where its numbers came from.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

from config import CONFIG, RAW_DIR, CalibrationConfig

__all__ = ["ensure_raw_dataset", "DownloadResult"]

_TIMEOUT_S = 120
_MIN_PLAUSIBLE_BYTES = 1_000_000  # the real file is ~45 MB; anything tiny is an error page


class DownloadResult:
    """Outcome of the fetch attempt, including why it failed if it did."""

    def __init__(self, path: Path | None, source_url: str | None, error: str | None = None):
        self.path = path
        self.source_url = source_url
        self.error = error

    @property
    def ok(self) -> bool:
        return self.path is not None and self.path.exists()

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        state = f"ok path={self.path}" if self.ok else f"failed error={self.error}"
        return f"<DownloadResult {state}>"


def _write_payload(content: bytes, target: Path, url: str) -> Path:
    """Persist the response, transparently unpacking a zip if we got one."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith((".xlsx", ".csv"))]
            if not members:
                raise ValueError(f"archive from {url} contains no xlsx/csv member")
            member = members[0]
            target = target.with_suffix(Path(member).suffix)
            target.write_bytes(archive.read(member))
        return target

    target.write_bytes(content)
    return target


def ensure_raw_dataset(cfg: CalibrationConfig | None = None, *, force: bool = False) -> DownloadResult:
    """Download the calibration dataset once and cache it under ``data/raw``.

    Returns a DownloadResult rather than raising, because an unavailable
    dataset is a supported (degraded) mode, not a crash.
    """
    cfg = cfg or CONFIG.calibration
    target = RAW_DIR / cfg.raw_filename

    if target.exists() and not force:
        return DownloadResult(target, source_url="cache")

    # Some mirrors serve the file under a different extension; accept either.
    for candidate in (target.with_suffix(".xlsx"), target.with_suffix(".csv")):
        if candidate.exists() and not force:
            return DownloadResult(candidate, source_url="cache")

    errors: list[str] = []
    for url in cfg.source_urls:
        try:
            response = requests.get(url, timeout=_TIMEOUT_S, headers={"User-Agent": "warehouse-flow-optimization/1.0"})
            response.raise_for_status()
            if len(response.content) < _MIN_PLAUSIBLE_BYTES:
                raise ValueError(
                    f"response too small ({len(response.content)} bytes) - likely an error page"
                )
            written = _write_payload(response.content, target, url)
            return DownloadResult(written, source_url=url)
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next mirror"
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")

    return DownloadResult(None, None, error="; ".join(errors))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    result = ensure_raw_dataset()
    if result.ok:
        size_mb = result.path.stat().st_size / 1e6
        print(f"OK  {result.path.name}  {size_mb:.1f} MB  (from {result.source_url})")
    else:
        print("FAILED to obtain calibration dataset:")
        print(result.error)
        print("\nThe pipeline will fall back to parametric defaults.")
