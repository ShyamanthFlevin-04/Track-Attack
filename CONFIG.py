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