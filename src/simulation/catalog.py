"""Synthetic SKU catalog, with popularity taken from the calibrated curve.

Two design decisions matter here:

1. **Popularity is assigned independently of category.** This is deliberate.
   If popular SKUs were also the urgent ones, classic ABC slotting would
   already be optimal and there would be nothing for an urgency-weighted
   policy to improve. Keeping them independent creates the realistic tension
   the project is actually about: a slow-moving SKU that always ships same-day
   may deserve a better slot than a fast-moving one that never does.

2. **Storage constraints come from physical attributes**, not from a random
   flag: cold-chain categories need cold slots, and the heaviest SKUs are
   restricted to ground level. This is what makes slot assignment a
   constrained problem rather than a simple sort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, CatalogConfig, DemandConfig
from src.calibration.fit_parameters import CalibrationParams

__all__ = ["build_catalog"]


def build_catalog(
    calibration: CalibrationParams,
    rng: np.random.Generator,
    catalog_cfg: CatalogConfig | None = None,
    demand_cfg: DemandConfig | None = None,
) -> pd.DataFrame:
    """Return one row per SKU.

    Columns
    -------
    sku_id            : ``SKU0001`` style identifier
    category          : product family
    is_cold, is_heavy : storage constraints
    volume_l, weight_kg, unit_value : physical / commercial attributes
    popularity        : share of total demand, from the calibrated curve
    express_propensity: relative likelihood of appearing in an urgent order
    """
    catalog_cfg = catalog_cfg or CONFIG.catalog
    demand_cfg = demand_cfg or CONFIG.demand
    n = catalog_cfg.n_skus

    sku_id = np.array([f"SKU{i + 1:04d}" for i in range(n)])

    category = rng.choice(
        catalog_cfg.categories, size=n, p=np.asarray(catalog_cfg.category_weights)
    )

    # Popularity from the real-data curve, then shuffled so that it carries no
    # relationship to category or to SKU id (see module docstring).
    popularity = calibration.popularity_weights(n)
    popularity = rng.permutation(popularity)

    # Physical attributes are log-uniform: most SKUs are small, a few are bulky.
    volume_l = np.exp(
        rng.uniform(np.log(catalog_cfg.volume_range_l[0]), np.log(catalog_cfg.volume_range_l[1]), n)
    )
    weight_kg = np.exp(
        rng.uniform(np.log(catalog_cfg.weight_range_kg[0]), np.log(catalog_cfg.weight_range_kg[1]), n)
    )
    unit_value = np.exp(
        rng.uniform(np.log(catalog_cfg.unit_value_range[0]), np.log(catalog_cfg.unit_value_range[1]), n)
    )

    # "Heavy" is the top slice of the weight distribution, so the constraint is
    # a physical consequence rather than an arbitrary label.
    heavy_cutoff = np.quantile(weight_kg, 1.0 - catalog_cfg.heavy_probability)
    is_heavy = weight_kg >= heavy_cutoff

    is_cold = np.isin(category, catalog_cfg.cold_categories)

    express_propensity = np.where(
        np.isin(category, demand_cfg.express_prone_categories),
        demand_cfg.express_prone_boost,
        1.0,
    )

    catalog = pd.DataFrame(
        {
            "sku_id": sku_id,
            "category": category,
            "is_cold": is_cold,
            "is_heavy": is_heavy,
            "volume_l": np.round(volume_l, 2),
            "weight_kg": np.round(weight_kg, 2),
            "unit_value": np.round(unit_value, 2),
            "popularity": popularity,
            "express_propensity": express_propensity,
        }
    )
    return catalog
