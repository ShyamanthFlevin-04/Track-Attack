# CONFIG.py
# ==========================================
# LKA System Configuration (OPTIMIZED FOR SPEED & STABILITY)
# ==========================================

# --- Feature Toggles ---
ENABLE_LANE_DETECTION = True
ENABLE_DRIVABLE_AREA  = False  # Disabled for maximum speed
SHOW_TELEMETRY        = True

# --- Calibration Defaults ---
DEFAULT_ROI_POINTS = [
    (0.40, 0.58),  # Top-Left
    (0.60, 0.58),  # Top-Right
    (1.00, 1.00),  # Bottom-Right
    (0.00, 1.00)   # Bottom-Left
]

# --- System Settings ---
WINDOW_WIDTH       = 1280
WINDOW_HEIGHT      = 720
MAIN_WINDOW_NAME   = "LKA System Output"
CALIB_WINDOW_NAME  = "Interactive Calibration (Drag Points)"

# --- Model Settings ---
MODEL_STRIDE  = 640
DEVICE_TARGET = '0'

# --- Performance Tuning ---
WARMUP_ROUNDS = 3
CV2_NUM_THREADS = 0
FRAME_QUEUE_SIZE = 4

# --- Log File ---
LOG_FILE_PATH = r"C:\Users\mailn\OneDrive\Desktop\Track-Attack\LKA_2026\Session_Log\session_log.txt"

# --- Stanley Controller & Network Settings ---
UDP_IP          = "172.27.192.1" 
UDP_PORT        = 5000           
STANLEY_K       = 1.0            # Reduced from 1.5/4.5 to eliminate death wobble
STANLEY_V_MPS   = 8.3            # Assumed speed (10 m/s = ~36 km/h)
MAX_STEER_RAD   = 0.8           # Max steering angle in radians

# ---------------------------------------------------------------------------
# Ego-Lane Stabilization — Dashed-Line Robustness
# ---------------------------------------------------------------------------
# Confidence decay multiplier applied each frame when a lane side is NOT detected.
# Range: (0, 1). Higher = slower decay (more memory). Default: 0.88 (~12 frames
# to fall from 100 → 20 confidence).
EGO_CONF_DECAY        = 0.88

# Confidence recovery blend weight when a lane IS freshly detected.
# Effective recovered conf = old_conf*(1-EGO_CONF_RECOVERY) + 100*EGO_CONF_RECOVERY.
# Range: (0, 1). Default: 0.50 (fast recovery but not instant).
EGO_CONF_RECOVERY     = 0.50

# Confidence threshold below which the phantom (persisted) track is
# abandoned and the side falls back to synthetic projection.
EGO_MIN_CONF          = 20.0

# Number of consecutive frames a new ego-lane candidate must differ from the
# current confirmed ego lane before the switch is accepted (hysteresis gate).
# Prevents oscillatory left-right flips at dashed-line boundaries. Default: 5.
EGO_SWITCH_HYSTERESIS = 5

# ---------------------------------------------------------------------------
# Ego-Lane Candidate Scoring Weights
# (proximity + heading_continuity + curvature_plausibility must sum to 1.0)
# If changed, ensure EGO_W_PROXIMITY + EGO_W_HEADING + EGO_W_CURVATURE == 1.0
# to keep scores in the 0–100 range; non-normalized weights are allowed but
# will shift the absolute scale of scores.
# ---------------------------------------------------------------------------
EGO_W_PROXIMITY  = 0.50   # Weight for lateral proximity to expected ego position
EGO_W_HEADING    = 0.30   # Weight for heading continuity with prior frame
EGO_W_CURVATURE  = 0.20   # Weight for curvature plausibility (penalise unrealistic kinks)

# ---------------------------------------------------------------------------
# Temporal Signal Smoothing (EMA)
# ---------------------------------------------------------------------------
# Alpha for smoothing final telemetry signals: heading_deg, curvature_k, radius_m.
# Lower alpha = smoother but laggier. Range: (0, 1). Default: 0.35.
TELEM_EMA_ALPHA  = 0.35