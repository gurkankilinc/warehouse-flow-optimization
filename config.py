"""Central configuration for the warehouse flow optimization project.

Every tunable parameter of the simulation, the models and the experiments lives
here so that a reader can understand (and change) the whole setup from a single
file. Values are grouped into dataclasses by concern.

Units used throughout the project:
    distance -> meters
    time     -> seconds (simulation clock), unless a field name says otherwise
    demand   -> units per SKU per day
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REPORTS_DIR = ROOT / "reports"

CALIBRATION_FILE = PROCESSED_DIR / "calibration.json"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Calibration (real-world data)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    """Where the real demand data comes from and how it is cleaned.

    The warehouse *geometry* is necessarily synthetic (no public dataset
    contains real slot layouts), but the *demand* side is calibrated against a
    real retail transaction dataset so that popularity, basket size and
    seasonality are estimated rather than invented.
    """

    # UCI Online Retail II: invoice-level transactions of a UK online retailer.
    # Chosen because it is the only easily obtainable public dataset that gives
    # BOTH basket composition (lines per order) and real calendar timestamps.
    source_name: str = "UCI Online Retail II"
    source_urls: tuple[str, ...] = (
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
    )
    raw_filename: str = "online_retail_II.xlsx"

    # Rows that are not real outbound demand and must be removed.
    cancelled_invoice_prefix: str = "C"
    # Stock codes that encode fees/adjustments rather than physical products.
    non_product_codes: tuple[str, ...] = (
        "POST", "DOT", "M", "m", "C2", "S", "BANK CHARGES", "AMAZONFEE",
        "PADS", "B", "CRUK", "D", "gift_0001", "TEST001", "TEST002",
    )
    min_unit_price: float = 0.01
    max_unit_price: float = 1000.0
    min_quantity: int = 1
    max_quantity: int = 5000

    # If the download fails the pipeline still runs, using these fallbacks.
    allow_synthetic_fallback: bool = True
    fallback_zipf_exponent: float = 1.15
    fallback_lines_per_order_mean: float = 12.0
    fallback_units_per_line_mean: float = 8.0


# --------------------------------------------------------------------------
# Warehouse layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WarehouseConfig:
    """A multi-block picker-to-parts warehouse.

    Layout: `n_aisles` parallel picking aisles, crossed by `n_cross_aisles`
    transverse aisles (front, middle, back). Racks line both sides of every
    picking aisle and the shipping dock sits on the front cross aisle.

    The number of cross aisles is not cosmetic -- it decides whether routing is
    a real problem. With a single cross aisle the tour length decomposes into
    an order-invariant term (every stop must be entered and exited from the
    same end) plus a one-dimensional traversal that sorting by aisle already
    solves optimally, so no routing heuristic can beat S-shape. Adding
    transverse aisles gives the picker a choice of exits and turns the tour
    into a genuinely combinatorial problem -- which is also why real
    distribution centres have them.
    """

    # Sized as a mid-size distribution centre: a 60 m x 33 m pick face area.
    # The scale matters -- in a small warehouse walking is a rounding error and
    # there is nothing for slotting or routing to win back. Here travel is a
    # realistic share of picker time, which is the regime the literature
    # describes and the one worth optimising.
    n_aisles: int = 20
    bays_per_aisle: int = 24
    levels: int = 3
    sides: int = 2  # racks on both sides of each aisle
    n_cross_aisles: int = 3  # front (with the dock), middle, back

    aisle_width: float = 3.0      # center-to-center distance between aisles (m)
    bay_depth: float = 1.2        # depth of one bay along the aisle (m)
    level_height: float = 1.6     # vertical distance between levels (m)
    cross_aisle_offset: float = 4.0  # distance from dock line to first bay (m)

    dock_x: float = 0.0           # dock sits at the west end of the cross aisle
    receiving_x: float = 0.0      # goods-in door; set equal to dock for v1

    # Storage zones. Cold items must go to cold slots; heavy items are
    # restricted to ground level for ergonomic reasons. These constraints make
    # the assignment problem genuinely constrained rather than trivial.
    cold_zone_aisles: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    heavy_max_level: int = 0      # heavy SKUs only on level 0

    @property
    def n_slots(self) -> int:
        return self.n_aisles * self.bays_per_aisle * self.sides * self.levels

    @property
    def aisle_length(self) -> float:
        """Distance from the front cross aisle to the far end of the racks."""
        return self.cross_aisle_offset + self.bays_per_aisle * self.bay_depth

    @property
    def cross_aisle_ys(self) -> tuple[float, ...]:
        """y positions of the transverse aisles. Index 0 carries the dock."""
        if self.n_cross_aisles <= 1:
            return (0.0,)
        step = self.aisle_length / (self.n_cross_aisles - 1)
        return tuple(i * step for i in range(self.n_cross_aisles))


# --------------------------------------------------------------------------
# Product catalog
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogConfig:
    # ~83% slot occupancy. Deliberately tight: if slots far outnumbered SKUs
    # every product would get a good location and the assignment problem would
    # be vacuous.
    n_skus: int = 2400

    categories: tuple[str, ...] = (
        "Electronics", "HomeLiving", "Apparel", "Food", "Beverage", "Office",
    )
    # Probability that a SKU belongs to each category (sums to 1).
    category_weights: tuple[float, ...] = (0.14, 0.24, 0.20, 0.16, 0.12, 0.14)
    # Which categories require cold storage.
    cold_categories: tuple[str, ...] = ("Food", "Beverage")
    # Probability a SKU is flagged heavy (ground level only).
    heavy_probability: float = 0.12

    volume_range_l: tuple[float, float] = (0.2, 8.0)      # liters per unit
    weight_range_kg: tuple[float, float] = (0.1, 25.0)
    unit_value_range: tuple[float, float] = (5.0, 900.0)


# --------------------------------------------------------------------------
# Demand and orders
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandConfig:
    """Length of the demand history used to train the forecasting model.

    History is generated analytically (fast) over a long horizon; the expensive
    discrete-event simulation only runs over the shorter evaluation window
    defined in SimulationConfig.
    """

    history_days: int = 540           # 18 months of daily demand
    start_date: str = "2023-01-02"    # a Monday, so weekday effects line up

    # Chosen so the warehouse runs at ~85% picker utilisation and misses a few
    # percent of its promises. That operating point is deliberate: below it
    # every policy scores 100% and the comparison is vacuous, and just above it
    # the queue collapses (at 225 orders/day on-time collapses to 28%). Real
    # warehouses live on exactly this cliff edge, which is why scheduling them
    # well is worth anything at all.
    base_orders_per_day: float = 200.0
    annual_growth: float = 0.12       # yearly trend
    noise_dispersion: float = 0.08    # gamma-Poisson overdispersion (CV ~0.28)

    # Weekday and month multipliers are NOT set here: they are estimated from
    # the real dataset by src.calibration.fit_parameters and read from
    # calibration.json, so the seasonality the model learns is a real pattern
    # rather than one we invented.

    # Hour-of-day arrival intensity (24 values, normalised internally).
    # Orders trickle in overnight and peak late morning and mid afternoon.
    hourly_arrival_weights: tuple[float, ...] = (
        0.20, 0.12, 0.08, 0.06, 0.06, 0.15, 0.45, 0.90,
        1.50, 2.10, 2.40, 2.20, 1.70, 1.85, 2.05, 1.95,
        1.60, 1.25, 1.00, 0.85, 0.70, 0.55, 0.40, 0.28,
    )

    # Service level mix: same-day is rare but drives urgency.
    service_levels: tuple[str, ...] = ("SAME_DAY", "NEXT_DAY", "STANDARD")
    service_level_weights: tuple[float, ...] = (0.15, 0.45, 0.40)
    # Categories skew towards faster service, which is what makes urgency
    # correlate with product identity (and therefore matter for slotting).
    express_prone_categories: tuple[str, ...] = ("Electronics", "Food")
    express_prone_boost: float = 2.0


# --------------------------------------------------------------------------
# Trucks and docks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TruckConfig:
    """Outbound truck schedule. Order deadlines are derived from this.

    deadline = truck departure - staging_buffer

    so a late truck relaxes its orders and an early one tightens them.
    """

    # Planned departure times (hour of day) for each daily wave.
    departure_hours: tuple[float, ...] = (10.0, 14.0, 18.0, 22.0)
    # A truck is expected to arrive this long before its departure.
    lead_time_before_departure_h: float = 2.0
    # Orders must be staged this long before departure to make the truck.
    staging_buffer_h: float = 0.5

    n_docks: int = 3
    loading_time_mean_h: float = 0.75
    loading_time_cv: float = 0.25

    # Arrival delay: mostly on time, occasionally very late (right-skewed).
    delay_probability: float = 0.35
    delay_mean_h: float = 0.8
    delay_shape: float = 1.6          # gamma shape; >1 gives a realistic tail
    early_probability: float = 0.15
    early_mean_h: float = 0.4

    truck_capacity_orders: int = 100

    # --- the second half of shipping: staging and loading --------------
    # Picking ends when an order reaches the staging area; it is not shipped
    # until a loader has physically put it on the vehicle. Modelling that as a
    # resourced process rather than an instant makes two things visible: a miss
    # can be caused by loading rather than by picking, and a staging area that
    # fills up pushes back on picking.
    # How many vehicles can be loaded at once. A crew works a truck *together*,
    # so this is a count of loading teams, not of people -- the per-order times
    # below are already a team's pace. Modelling it the other way round (one
    # person per truck) makes loading the dominant bottleneck: the crew then
    # burns 100% of every departure window, manages ~50 orders a truck against
    # ~200 arriving a day, and the staging lanes fill until picking stops.
    n_load_teams: int = 2
    load_setup_per_order_s: float = 14.0    # position, scan, secure
    load_per_unit_s: float = 0.05           # moving staged cases onto the deck
    # How much of the pre-departure window the crew can actually use.
    min_loading_window_h: float = 0.25
    # Orders that may wait on the staging lanes at once. When it is full,
    # pickers cannot drop off and outbound flow stalls -- which is exactly what
    # happens on a real floor when the dock backs up.
    staging_capacity_orders: int = 300

    # Minimum time between an order arriving and the truck it can leave on,
    # per service level (same order as DemandConfig.service_levels). This is
    # what turns a service level into a concrete, tight-or-loose deadline.
    service_min_lead_h: tuple[float, ...] = (1.5, 12.0, 30.0)


# --------------------------------------------------------------------------
# Picking operation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PickingConfig:
    n_pickers: int = 4
    n_putaway_workers: int = 2
    walking_speed_mps: float = 1.15

    # Time model for a pick, in seconds.
    setup_time_s: float = 45.0        # per picking tour (get cart, scan, etc.)
    per_line_time_s: float = 12.0
    # Units arrive in cases, so per-unit handling is fast. Keeping this low is
    # also what keeps travel a meaningful share of picker time -- the share
    # actually measured and reported by the experiments.
    per_unit_time_s: float = 1.0
    level_penalty_s: float = 7.0      # per level above ground
    drop_off_time_s: float = 60.0     # staging the completed tour

    # Pallet-truck picking: a tour holds enough for roughly two average orders,
    # which is what makes order batching a decision worth optimising rather
    # than a formality.
    max_lines_per_tour: int = 45
    max_volume_per_tour_l: float = 800.0

    # ---- Hidden factors -------------------------------------------------
    # These perturb actual pick times but are NEVER exposed as model features.
    # They guarantee irreducible error, so the pick-time model cannot simply
    # re-learn the simulator's own formula and score perfectly.
    congestion_base: float = 1.0
    congestion_amplitude: float = 0.22    # time-of-day congestion swing
    congestion_noise_sd: float = 0.10
    picker_skill_sd: float = 0.12         # per-picker speed multiplier spread
    pick_time_noise_cv: float = 0.16      # residual lognormal noise


# --------------------------------------------------------------------------
# Simulation run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42

    # The discrete-event simulation is expensive, so it runs on a window at the
    # end of the demand history rather than the full 18 months. That window is
    # split into three consecutive phases:
    #
    #   [ training_days ][ warmup_days ][ evaluation_days ]
    #
    # training   a baseline run over these days produces the tour-level data
    #            used to fit the pick-time and SLA-risk models. Those models
    #            never see the evaluation period, so there is no leakage.
    # warmup     discarded, so the measured period does not inherit a queue
    #            state that depends on how training ended.
    # evaluation the only days that count towards reported KPIs.
    training_days: int = 60
    warmup_days: int = 7
    evaluation_days: int = 42

    @property
    def window_days(self) -> int:
        return self.training_days + self.warmup_days + self.evaluation_days

    @property
    def measure_start_s(self) -> float:
        return (self.training_days + self.warmup_days) * 86400.0

    @property
    def measure_end_s(self) -> float:
        return self.window_days * 86400.0

    @property
    def training_end_s(self) -> float:
        return self.training_days * 86400.0

    shift_start_hour: float = 6.0
    shift_end_hour: float = 23.0

    # Slotting is decided once, at the start of the measured period, from the
    # forecast for the weeks ahead. Rolling re-slotting is left out on purpose:
    # relocating stock costs picker time, and modelling the benefit without
    # modelling that cost would overstate the gain.


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    # Walk-forward backtest for the demand model. Each fold scores a block of
    # this many days, but with lags of realised demand, so the backtest is
    # one-day-ahead accuracy, not a 7-day-ahead forecast. Also the length of the
    # early-stopping validation tail.
    forecast_horizon_days: int = 7
    n_backtest_folds: int = 4
    min_train_days: int = 180

    demand_lags: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
    demand_rolling_windows: tuple[int, ...] = (7, 14, 28, 56)

    # Gradient boosting hyperparameters (shared defaults; kept modest so the
    # whole pipeline runs in minutes on a laptop CPU).
    n_estimators: int = 400
    learning_rate: float = 0.06
    max_depth: int = 7
    min_samples_leaf: int = 40
    early_stopping_rounds: int = 40

    # SLA risk classifier decision threshold is tuned on validation data, but
    # this is the fallback if tuning is skipped.
    default_risk_threshold: float = 0.5


# --------------------------------------------------------------------------
# Decision policies
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Knobs of the decision layer -- the part the project actually optimises."""

    # --- slotting -----------------------------------------------------
    # How much history the baseline ABC policy looks back over.
    abc_lookback_days: int = 90
    # Urgency weight applied on top of expected pick frequency:
    #     weight = 1 + urgency_beta * (share of picks in same-day orders)
    # With beta = 0 the optimised policy collapses to frequency-only slotting,
    # which is what the ablation uses to isolate the urgency contribution.
    urgency_beta: float = 2.5
    # Solve the slot assignment exactly (Hungarian) instead of greedily.
    # Greedy is provably optimal without constraints and near-optimal with
    # them; the exact solver exists so tests can verify that claim.
    use_exact_assignment: bool = False

    # --- release ------------------------------------------------------
    # An order is not eligible for picking until its truck is within this many
    # hours. This is wave release, and it is the handshake between the two
    # halves of shipping: without it pickers run ahead on orders whose truck
    # leaves tomorrow, fill the staging lanes, and block themselves out of the
    # work for the truck that is actually at the door. (The simulation
    # deadlocks outright if you leave it out -- which is a fair model of what
    # happens to a real floor when nobody controls order release.)
    # Chosen by sweeping it: at 8 h the on-time rate collapses to 53% because
    # all the work bunches into the hours before each departure, and at 36 h
    # orders sit on the staging lanes for 16 hours, which is using the dock as
    # a warehouse. 24 h keeps service near the achievable ceiling and splits
    # the remaining misses roughly evenly between picking and loading -- which
    # is the honest picture of a warehouse constrained at both ends.
    release_horizon_h: float = 24.0

    # --- dispatch -----------------------------------------------------
    # Orders whose predicted miss-probability exceeds this are escalated to the
    # front of the queue regardless of their slack.
    risk_escalation_threshold: float = 0.35
    # Safety margin added to predicted pick time when computing slack.
    slack_safety_margin_s: float = 300.0

    # --- batching -----------------------------------------------------
    # Candidate pool the savings-based batcher considers when growing a tour.
    batching_candidate_pool: int = 25
    # Orders are only batched together if their deadlines are within this gap,
    # so batching never sacrifices an urgent order for a relaxed one.
    batching_max_deadline_gap_h: float = 4.0


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    # Independent replications per scenario, so reported deltas come with a
    # confidence interval instead of being a single lucky run.
    n_replications: int = 5
    replication_seed_offset: int = 1000

    scenarios: tuple[str, ...] = (
        "baseline",
        "slotting_only",
        "dispatch_only",
        "routing_only",
        "optimized",
    )

    stress_tests: tuple[str, ...] = (
        "demand_surge",
        "truck_delay_heavy",
        "picker_shortage",
    )
    demand_surge_multiplier: float = 1.4
    truck_delay_multiplier: float = 2.5
    # Must be below PickingConfig.n_pickers or the "shortage" changes nothing --
    # a mistake that silently produced a stress result identical to the normal
    # one until the two columns were compared.
    picker_shortage_count: int = 3


# --------------------------------------------------------------------------
# Root config object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    truck: TruckConfig = field(default_factory=TruckConfig)
    picking: PickingConfig = field(default_factory=PickingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CONFIG = Config()
