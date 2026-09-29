"""Every default number UrbanFlow uses, grouped by topic, each citing its plan section.

Engine and model code import numbers from here instead of writing literals, so a value
is defined, documented and tested once. Units are SI (m, s, m/s, m/s², kg, W) unless noted.
This module is pure stdlib so that importing it is free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# --------------------------------------------------------------------------- numerics
TIME_EPS: Final = 1e-9  # s; stage ends and step emission compare with -1e-9 (H.2, E.7 §1.3)
GEOM_EPS: Final = 1e-9  # dimensionless parameter tolerance in geometry predicates (E.5)
POSITION_EPS: Final = 1e-6  # m; on-link / overlap tolerance of invariants I3, I4 (F.7)
SECONDS_PER_HOUR: Final = 3600.0  # veh/h <-> veh/s conversions (E.7 §1.3, J.2)

# --------------------------------------------------------------------------- simulation config
DT: Final = 1.0  # s; default step (AB 6.1)
DT_MIN: Final = 0.1  # s; hard lower bound of dt (AB 6.1)
DT_MAX: Final = 2.0  # s; hard upper bound; the ballistic update is stable up to 2 s (G.3)
DT_RECOMMENDED_MIN: Final = 0.2  # s; W701 outside [0.2, 1.0] (E.8)
DT_RECOMMENDED_MAX: Final = 1.0  # s; W701 (E.8)
DURATION: Final = 3600.0  # s; default simulated duration (AB 6.1)
SEED_MAX: Final = 2**63 - 1  # largest root seed (AB 6.1)
HALTING_SPEED: Final = 0.1  # m/s; a vehicle is halting below this (AB 6.1, J.1)
DEADLOCK_TIMEOUT: Final = 300.0  # s; watchdog teleport after this long stuck; 0 = off (F.3)
REROUTE_PERIOD: Final = 60.0  # s; dynamic router refresh period (AB 6.1, I.1)
METRICS_INTERVAL_S: Final = 10.0  # s; timeseries sample interval (AG.1 R9, B.2 #8)
RECORD_CHUNK_ROWS: Final = 1_000_000  # rows per replay chunk (K.3, B.2 #8)
RECORD_MAX_CHUNK_STEPS: Final = 300  # frames per replay chunk (K.3, B.2 #8)
RECORD_COMPRESSLEVEL: Final = 1  # deflate level of replay chunks (AB 6.1)
RECORD_COMPRESSLEVEL_MAX: Final = 9  # zlib maximum

# --------------------------------------------------------------------------- vehicle model (G.7)
VEHICLE_LENGTH: Final = 5.0  # m; built-in car (E.7 VehicleType)
VEHICLE_WIDTH: Final = 1.8  # m; built-in car
VEHICLE_MAX_SPEED: Final = 36.1  # m/s; built-in car; effective v0 = min(max_speed, f·limit)
IDM_ACCEL: Final = 1.5  # m/s²; IDM a, typical city value (Treiber & Kesting 2013)
IDM_DECEL: Final = 2.0  # m/s²; IDM b, comfortable deceleration
IDM_EMERGENCY_DECEL: Final = 6.0  # m/s²; b_emerg, physical braking bound, assumption A1 (G.2)
IDM_MIN_GAP: Final = 2.0  # m; s0
IDM_HEADWAY: Final = 1.5  # s; T; saturation ~1800 veh/h/lane at 13.9 m/s (G.7)
IDM_DELTA: Final = 4.0  # free-road exponent δ (model_params.delta)
IDM_OVERLAP_GAP: Final = 0.01  # m; at or below this gap a_IDM = -b_emerg (G.1)
BALLISTIC_FLOOR: Final = 1e-9  # m, m/s, m/s²; G.3 floors of C and b*, float noise of its tests
SPEED_FACTOR_MEAN: Final = 1.0  # speed factor f ~ N(1.0, 0.1) clipped to [0.8, 1.2] (G.7)
SPEED_FACTOR_STD: Final = 0.1
SPEED_FACTOR_MIN: Final = 0.8
SPEED_FACTOR_MAX: Final = 1.2
MOBIL_POLITENESS: Final = 0.2  # p (Kesting, Treiber & Helbing 2007)
MOBIL_THRESHOLD: Final = 0.1  # m/s²; Δa_th
MOBIL_SAFE_DECEL: Final = 4.0  # m/s²; b_safe
SAFETY_MARGIN: Final = 0.5  # m; s_m of the safe-speed cap (G.2)
LOOKAHEAD_MIN_DISTANCE: Final = 200.0  # m; leader lookahead floor (F.1 step 4)
ANTICIPATION_DECEL_FACTOR: Final = 0.5  # speed-limit anticipation brakes when r > 0.5 b (G.1)
MIN_EFFECTIVE_SPEED: Final = 1.0  # m/s; floor in d/max(v, 1) orderings and L/max(v̄, 1) (F.3, I.1)

# Built-in vehicle types as field overrides of the defaults above (E.7 "Built-in types").
BUILTIN_VEHICLE_TYPES: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "car": MappingProxyType({}),
        "bus": MappingProxyType(
            {
                "vclass": "bus",
                "length": 12.0,
                "width": 2.55,
                "max_speed": 25.0,
                "accel": 1.0,
                "decel": 1.5,
                "min_gap": 2.5,
                "speed_factor": MappingProxyType({"std": 0.05}),
            }
        ),
        "truck": MappingProxyType(
            {
                "vclass": "truck",
                "length": 12.0,
                "width": 2.5,
                "max_speed": 25.0,
                "accel": 0.8,
                "decel": 1.5,
                "min_gap": 2.5,
            }
        ),
        "emergency": MappingProxyType(
            {
                "vclass": "emergency",
                "max_speed": 41.7,
                "accel": 2.5,
                "speed_factor": MappingProxyType({"mean": 1.3, "std": 0.0, "min": 1.3, "max": 1.3}),
            }
        ),
    }
)

# Vehicle type field bounds (E.7 VehicleType table).
VEHICLE_LENGTH_MAX: Final = 30.0  # m; length in (0, 30]
VEHICLE_WIDTH_MAX: Final = 4.0  # m; width in (0, 4]
VEHICLE_ACCEL_MAX: Final = 10.0  # m/s²; accel and decel in (0, 10]
EMERGENCY_DECEL_MAX: Final = 15.0  # m/s²; decel <= emergency_decel <= 15
MIN_GAP_MAX: Final = 20.0  # m; min_gap in [0, 20]
HEADWAY_MAX: Final = 10.0  # s; headway in (0, 10]
LC_THRESHOLD_MAX: Final = 5.0  # m/s²; lc_threshold in [0, 5]
LC_SAFE_DECEL_MAX: Final = 15.0  # m/s²; lc_safe_decel in (0, 15]

# --------------------------------------------------------------------------- intersections (F.3)
DECISION_MARGIN: Final = 5.0  # m; D_i = v²/(2b) + v·dt + DECISION_MARGIN
GAP_ACCEPT_MARGIN: Final = 1.0  # s; τ widening every occupancy window
ETA_END_ACCEL_FACTOR: Final = 0.5  # late-ETA acceleration = factor · a
YELLOW_MAX_DECEL: Final = 3.0  # m/s²; ITE-style dilemma-zone threshold (F.3, G.7)
STOP_LINE_CLEARANCE: Final = 0.5  # m; g_obs = d + s0 - 0.5 stops the front 0.5 m before a line

# --------------------------------------------------------------------------- lane changing (G.4)
LC_NO_DISCRETIONARY_ZONE: Final = 20.0  # m; no discretionary change closer to the stop line
LC_MANDATORY_BIAS: Final = 1.0  # m/s²; urgency scale of mandatory changes
LC_MANDATORY_DISTANCE: Final = 200.0  # m; D_m in β_m = min(10, bias·D_m/max(d, 1))
LC_MANDATORY_BIAS_MAX: Final = 10.0  # m/s²; cap of β_m
LC_URGENCY_MIN_DISTANCE: Final = 1.0  # m; floor of d in β_m
LC_POLITENESS_OFF_DISTANCE: Final = 50.0  # m; p = 0 for mandatory changes closer than this
LC_COOLDOWN: Final = 3.0  # s; minimum time between discretionary changes
LC_VISUAL_DURATION: Final = 2.0  # s; lat_offset decays at w / duration (render only)
LC_KEEP_RIGHT_BIAS: Final = 0.0  # m/s²; keep-right bias, off by default
LC_CHECK_INTERVAL: Final = 1.0  # s; discretionary evaluation subsampling (S.1)
LC_SORT_KEY_SCALE: Final = 1e6  # m; combined neighbour key link·K + pos (G.4)

# --------------------------------------------------------------------------- compiler (E.5)
MAX_VEHICLE_WIDTH: Final = 2.6  # m; widest vehicle assumed by conflict zones
LATERAL_MARGIN: Final = 0.4  # m; clearance added to MAX_VEHICLE_WIDTH
CONFLICT_WIDTH: Final = MAX_VEHICLE_WIDTH + LATERAL_MARGIN  # m; W_max of d_sep and zone sizing
TURN_LATERAL_ACCEL: Final = 2.0  # m/s²; connector limit sqrt(a_lat · R_min) (E.5, G.7)
MIN_LANE_LENGTH: Final = 5.0  # m; shorter trimmed lanes raise E802
MITER_LIMIT: Final = 4.0  # cap of the offset miter factor 1/cos(φ/2); bevel beyond it
CONNECTOR_SAMPLE_ANGLE: Final = math.pi / 16  # rad; n = max(2, ceil(θ/(π/16)) + 1)
CONNECTOR_MIN_SAMPLES: Final = 2  # points of a straight connector
LOW_ANGLE_CROSSING: Final = math.radians(15.0)  # rad; below it crossing zones are sampled
SEPARATION_SAMPLE_STEP: Final = 0.25  # m; resolution of the d_sep sampler (linearly refined)
LINK_BASE_GAP: Final = 1.0  # m; link_base[l+1] = link_base[l] + L_l + gap (E.3)
DETECTOR_LENGTH: Final = 30.0  # m; lane_detector_start = L - 30 (E.3), actuated default (H.4)

# --------------------------------------------------------------------- render geometry (E.5 #7)
INTERSECTION_DISC_POINTS: Final = 16  # samples of the radius disc in intersection polygons
RENDER_COORD_DECIMALS: Final = 3  # render geometry coordinates are rounded to 1 mm

# --------------------------------------------------------------------------- derivation (E.7)
LANE_WIDTH: Final = 3.2  # m; network default lane width
LANE_WIDTH_MIN: Final = 2.0  # m
LANE_WIDTH_MAX: Final = 5.0  # m
SPEED_LIMIT: Final = 13.89  # m/s (50 km/h); network default speed limit
SPEED_LIMIT_MIN: Final = 1.0  # m/s
SPEED_LIMIT_MAX: Final = 70.0  # m/s; also the max_speed bound of vehicle types
SETBACK: Final = 2.0  # m; derived radius = max incident road width + SETBACK (E.7 §1.5)
INTERSECTION_RADIUS_MAX: Final = 200.0  # m
STRAIGHT_MAX_ANGLE: Final = math.radians(45.0)  # rad; |Δ| <= 45° can be straight (§1.5 step 5)
UTURN_MIN_ANGLE: Final = math.radians(135.0)  # rad; U-turn needs |Δ| >= 135°
OPPOSING_MIN_ANGLE: Final = math.radians(135.0)  # rad; approaches pair into an axis (§1.5 step 8)
STOP_DWELL: Final = 20.0  # s; default transit dwell (E.7 Stop)
STOP_END_CLEARANCE: Final = 1.0  # m; stop position <= lane length - 1 m (E807)
STOP_NEAR_LINE_DISTANCE: Final = 50.0  # m; W306 threshold

# --------------------------------------------------------------------------- signals (E.7, H)
PHASE_DURATION: Final = 30.0  # s; default fixed-time green
PHASE_DURATION_MAX: Final = 600.0  # s
SIGNAL_YELLOW: Final = 3.0  # s
SIGNAL_ALL_RED: Final = 1.0  # s
SIGNAL_INTERGREEN_MAX: Final = 10.0  # s; yellow and all_red in [0, 10]
DEFAULT_MIN_GREEN_S: Final = 5.0  # s; signal min_green default, RL min-green (AG.1 R9)
MIN_GREEN_MAX: Final = 120.0  # s
SIGNAL_MAX_GREEN: Final = 60.0  # s
MAX_GREEN_MAX: Final = 600.0  # s
ACTUATED_GAP: Final = 3.0  # s; gap-out threshold (H.4)
MAX_PRESSURE_DECISION_INTERVAL: Final = 5.0  # s (H.4)
WEBSTER_SATURATION_FLOW: Final = 1800.0  # veh/h/lane (H.4)
WEBSTER_CYCLE_MIN: Final = 30.0  # s; cycle_bounds default (H.4)
WEBSTER_CYCLE_MAX: Final = 180.0  # s
WEBSTER_LOST_TIME_FACTOR: Final = 1.5  # C0 = (1.5 L + 5) / (1 - Y) (Webster 1958)
WEBSTER_CYCLE_CONSTANT: Final = 5.0  # s
WEBSTER_Y_MAX: Final = 0.95  # Y >= 0.95 -> C = C_max
PREEMPTION_DETECTION_DISTANCE: Final = 150.0  # m (H.4)

# --------------------------------------------------------------------------- routing (I.1)
DYNAMIC_EWMA_ALPHA: Final = 0.3  # travel-time EWMA weight
REROUTE_HYSTERESIS: Final = 0.1  # replace the route only if >10% cheaper

# --------------------------------------------------------------------------- metrics / RL (AG.1 R9)
QUEUE_FRONT_TOLERANCE_M: Final = 10.0  # m; a queue must start within 10 m of the stop line (J.3)
VEH_SPACING_REF_M: Final = 7.5  # m; jam spacing, sumo-rl parity (5 m length + 2.5 m gap)
APPROACH_DISTANCE_M: Final = 100.0  # m; "approaching" vehicles window (H.3, L.4)
WAIT_REWARD_SCALE_S: Final = 100.0  # s; waiting-time reward normaliser (L.5)
BRAKE_FLAG_DECEL: Final = 0.5  # m/s²; frame braking flag when a < -0.5 (K.10.1)
TIME_IN_PHASE_REF_S: Final = 60.0  # s; time-in-phase observation normaliser (L.4)
SUMMARY_PERCENTILE: Final = 95.0  # %; the travel_time.p95 summary statistic (J.6)

# --------------------------------------------------------------------------- facade / CLI runs
PROGRESS_REFRESH_S: Final = 0.1  # s wall time between progress updates of run() (AA 5.3)
RUN_ID_SLUG_MAX: Final = 40  # characters of the scenario-name slug in a run id (AC 7.2)
RUN_ID_SUFFIX_BYTES: Final = 2  # random bytes -> 4 hex characters ending a run id (AC 7.2)


# --------------------------------------------------------------------------- energy proxy (G.9)
@dataclass(frozen=True, slots=True)
class EnergyParams:
    """Per-vclass energy proxy parameters (uncalibrated proxy, G.9)."""

    mass_kg: float
    cda_m2: float  # drag coefficient times frontal area
    rolling: float  # C_r
    efficiency: float  # η, tank-to-wheel
    idle_kw: float  # P_idle while halting
    fuel: str  # key of CO2_G_PER_MJ


GRAVITY: Final = 9.81  # m/s²; standard gravity
AIR_DENSITY: Final = 1.2  # kg/m³; air at ~20 °C, sea level
ENERGY_DEFAULTS: Final[Mapping[str, EnergyParams]] = MappingProxyType(
    {
        "car": EnergyParams(1500.0, 0.70, 0.010, 0.25, 1.0, "gasoline"),
        "bus": EnergyParams(12000.0, 6.0, 0.008, 0.35, 3.0, "diesel"),
        "truck": EnergyParams(15000.0, 6.5, 0.008, 0.35, 3.0, "diesel"),
        "emergency": EnergyParams(2500.0, 1.0, 0.010, 0.25, 1.5, "gasoline"),
    }
)
# g CO2 per MJ of fuel energy, IPCC motor gasoline / diesel oil (G.9)
CO2_G_PER_MJ: Final[Mapping[str, float]] = MappingProxyType({"gasoline": 69.3, "diesel": 74.1})
J_PER_KWH: Final = 3.6e6  # J in one kWh
J_PER_MJ: Final = 1e6

# ------------------------------------------------------------------ scenario limits (E.7, E.8)
MAX_SCENARIO_BYTES: Final = 64 * 1024 * 1024  # E013 file size cap
MAX_JSON_DEPTH: Final = 64  # E000: deeper nesting (pydantic serialises ~100 levels)
MAX_ID_LENGTH: Final = 128  # Id pattern length
MAX_COORDINATE: Final = 1e7  # m; abs(coord) bound
MAX_INTERSECTIONS: Final = 200_000
MAX_ROADS: Final = 500_000
MAX_ROAD_POINTS: Final = 10_000
MAX_LANES_PER_ROAD: Final = 16
MAX_PHASES: Final = 32
MAX_VEHICLE_TYPES: Final = 256
MAX_ROUTE_CHOICES: Final = 64  # FlowSpec.routes entries
MAX_NAME_LENGTH: Final = 100  # meta.name
MAX_DESCRIPTION_LENGTH: Final = 10_000  # meta.description
MAX_TAGS: Final = 32  # meta.tags
MAX_STREET_NAME_LENGTH: Final = 200  # road.name
SHORT_HASH_LENGTH: Final = 12  # hex characters of Scenario.short_hash (AA 5.2)
MESSAGE_DECIMALS: Final = 6  # computed floats quoted in issue messages are rounded (E.8)

# --------------------------------------------------------------------------- generators (E.10)
RATIO_SUM_TOLERANCE: Final = 1e-9  # turn ratios must sum to 1 within this
SINGLE_ARM_LENGTH: Final = 250.0  # m; single_intersection arm length
SINGLE_DEMAND_RATE: Final = 600.0  # veh/h per approach
SINGLE_TURN_RATIOS: Final = (0.15, 0.7, 0.15)  # (far, straight, near) turn shares
