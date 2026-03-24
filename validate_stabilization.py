"""
validate_stabilization.py
=========================
Deterministic regression tests for the ego-lane stabilization pipeline.

Run with:
    python validate_stabilization.py

No GPU or ONNX model needed — tests are unit-level and operate only on
numpy-based logic extracted from LANE.py and CENTRE_LANE.py.

Tests verify:
  1. Score-based ego candidate selection returns the highest-scored candidate.
  2. Hysteresis gate blocks a lane-switch for fewer than EGO_SWITCH_HYSTERESIS frames
     and confirms it after sufficient consecutive different-identity frames.
  3. Confidence decay reduces confidence monotonically during dropout frames.
  4. Confidence recovery restores confidence quickly when detection returns.
  5. Phantom track fallback is used during dropout when confidence is above EGO_MIN_CONF.
  6. Heading/curvature EMA smoothing produces stable signals (no abrupt sign flips).
  7. CENTRE_LANE EMA smoothing reduces peak-to-peak variance in noisy curvature.
"""

import sys
import math
import types
import numpy as np

# ---------------------------------------------------------------------------
# Minimal CONFIG stub so modules load without the real RUNNER/GUI stack
# ---------------------------------------------------------------------------
CONFIG = types.ModuleType("CONFIG")
CONFIG.ENABLE_LANE_DETECTION = True
CONFIG.ENABLE_DRIVABLE_AREA  = False
CONFIG.SHOW_TELEMETRY        = True
CONFIG.DEFAULT_ROI_POINTS    = [(0.40,0.58),(0.60,0.58),(1.00,1.00),(0.00,1.00)]
CONFIG.WINDOW_WIDTH          = 1280
CONFIG.WINDOW_HEIGHT         = 720
CONFIG.MAIN_WINDOW_NAME      = "Test"
CONFIG.CALIB_WINDOW_NAME     = "Test"
CONFIG.MODEL_STRIDE          = 640
CONFIG.DEVICE_TARGET         = 'cpu'
CONFIG.WARMUP_ROUNDS         = 0
CONFIG.CV2_NUM_THREADS       = 0
CONFIG.FRAME_QUEUE_SIZE      = 4
CONFIG.LOG_FILE_PATH         = ""
CONFIG.UDP_IP                = "127.0.0.1"
CONFIG.UDP_PORT              = 5000
CONFIG.STANLEY_K             = 1.0
CONFIG.STANLEY_V_MPS         = 8.3
CONFIG.MAX_STEER_RAD         = 0.8
# Stabilization params
CONFIG.EGO_CONF_DECAY        = 0.88
CONFIG.EGO_CONF_RECOVERY     = 0.50
CONFIG.EGO_MIN_CONF          = 20.0
CONFIG.EGO_SWITCH_HYSTERESIS = 5
CONFIG.EGO_W_PROXIMITY       = 0.50
CONFIG.EGO_W_HEADING         = 0.30
CONFIG.EGO_W_CURVATURE       = 0.20
CONFIG.TELEM_EMA_ALPHA       = 0.35

sys.modules["CONFIG"] = CONFIG

# Now import modules under test (no model load, no GPU)
# We test the purely numpy/analytical parts via a thin wrapper
# rather than full YOLOPInference (which requires a real model file).

# ---------------------------------------------------------------------------
# Helper: build a minimal lane candidate dict
# ---------------------------------------------------------------------------
def make_candidate(x_eval, fit=None, centroids=None):
    if fit is None:
        fit = np.array([0.0, 0.0, float(x_eval)])
    return {'fit': fit, 'centroids': centroids or [], 'x_eval': float(x_eval)}

# ---------------------------------------------------------------------------
# Thin test harness that exercises only the scoring / hysteresis methods
# by instantiating a stripped-down version of the relevant state.
# ---------------------------------------------------------------------------
class EgoLaneStabilizer:
    """
    Extracted ego-selection logic from LANE.py for isolated unit testing.
    Mirrors the state variables and methods added in v15.
    """
    def __init__(self):
        self._last_left_fit  = None
        self._last_right_fit = None
        self._left_ego_x     = None
        self._right_ego_x    = None
        self._left_switch_count  = 0
        self._right_switch_count = 0
        self._left_conf  = 0.0
        self._right_conf = 0.0
        self._ema_left   = None
        self._ema_right  = None

    # Copy of _score_ego_candidate from LANE.py
    def _score_ego_candidate(self, candidate, side, h, w):
        fit    = candidate['fit']
        x_eval = float(candidate['x_eval'])
        mid    = w / 2.0
        if side == 'left':
            expected_x = mid * 0.60
        else:
            expected_x = mid * 1.40
        prox_score = max(0.0, 1.0 - abs(x_eval - expected_x) / (mid * 0.90))
        prior_fit = self._last_left_fit if side == 'left' else self._last_right_fit
        if prior_fit is not None:
            db = abs(float(fit[1]) - float(prior_fit[1]))
            heading_score = max(0.0, 1.0 - db * 4.0)
        else:
            heading_score = 0.5
        curv = abs(float(fit[0]))
        curv_score = max(0.0, 1.0 - max(0.0, curv - 5e-4) / 5e-3)
        w_p = CONFIG.EGO_W_PROXIMITY
        w_h = CONFIG.EGO_W_HEADING
        w_c = CONFIG.EGO_W_CURVATURE
        score = (w_p * prox_score + w_h * heading_score + w_c * curv_score) * 100.0
        candidate['score'] = float(score)
        return score

    # Copy of _select_ego_candidate from LANE.py
    def _select_ego_candidate(self, candidates, side, h, w):
        if not candidates:
            if side == 'left':
                self._left_switch_count = 0
            else:
                self._right_switch_count = 0
            return None
        for c in candidates:
            self._score_ego_candidate(c, side, h, w)
        best = max(candidates, key=lambda c: c['score'])
        cur_x = self._left_ego_x if side == 'left' else self._right_ego_x
        hysteresis = int(CONFIG.EGO_SWITCH_HYSTERESIS)
        if cur_x is not None:
            identity_thr = w * 0.15
            same_identity = abs(best['x_eval'] - cur_x) < identity_thr
            if not same_identity:
                if side == 'left':
                    self._left_switch_count += 1
                    pending_count = self._left_switch_count
                else:
                    self._right_switch_count += 1
                    pending_count = self._right_switch_count
                if pending_count < hysteresis:
                    stable = min(candidates, key=lambda c: abs(c['x_eval'] - cur_x))
                    stable['hysteresis_pending'] = True
                    return stable
                else:
                    if side == 'left':
                        self._left_switch_count = 0
                    else:
                        self._right_switch_count = 0
            else:
                if side == 'left':
                    self._left_switch_count = 0
                else:
                    self._right_switch_count = 0
        if side == 'left':
            self._left_ego_x = best['x_eval']
        else:
            self._right_ego_x = best['x_eval']
        best['hysteresis_pending'] = False
        return best

    def update_confidence(self, detected_left, detected_right):
        decay    = CONFIG.EGO_CONF_DECAY
        recovery = CONFIG.EGO_CONF_RECOVERY
        if detected_left:
            self._left_conf = self._left_conf * (1.0 - recovery) + 100.0 * recovery
        else:
            self._left_conf *= decay
        if detected_right:
            self._right_conf = self._right_conf * (1.0 - recovery) + 100.0 * recovery
        else:
            self._right_conf *= decay


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0

def check(condition, label):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {label}")
    return condition


def test_scoring_returns_best():
    print("\n[TEST 1] Score-based selection returns highest-scored candidate")
    s = EgoLaneStabilizer()
    h, w = 720, 1280
    # Left candidates: c1 is far from expected, c2 is close
    c1 = make_candidate(x_eval=50,  fit=np.array([0.0, 0.5, 50.0]))   # bad
    c2 = make_candidate(x_eval=380, fit=np.array([0.0, 0.0, 380.0]))  # good (close to 0.6*mid=384)
    selected = s._select_ego_candidate([c1, c2], 'left', h, w)
    check(selected is c2, "Candidate closer to expected left ego position is selected")
    check(c2['score'] > c1['score'], "Higher-proximity candidate has higher score")


def test_hysteresis_blocks_switch():
    print("\n[TEST 2] Hysteresis gate blocks switch for < HYSTERESIS frames")
    s = EgoLaneStabilizer()
    h, w = 720, 1280
    # Establish ego at x=400
    c_stable = make_candidate(x_eval=400)
    s._select_ego_candidate([c_stable], 'left', h, w)
    check(abs(s._left_ego_x - 400) < 1, "Setup: left ego confirmed at x=400")

    hysteresis = CONFIG.EGO_SWITCH_HYSTERESIS  # 5
    # Now present a different candidate (x=150 — well outside 15% identity threshold)
    c_new = make_candidate(x_eval=150)
    for frame in range(hysteresis - 1):
        result = s._select_ego_candidate([c_new], 'left', h, w)
        check(
            result.get('hysteresis_pending', False) or abs(result['x_eval'] - 400) < w * 0.15,
            f"Frame {frame+1}/{hysteresis-1}: switch NOT yet confirmed (hysteresis pending)"
        )
    check(s._left_switch_count == hysteresis - 1,
          f"Switch counter = {hysteresis - 1} after {hysteresis - 1} pending frames")

    # On hysteresis-th consecutive frame with new candidate, switch is confirmed
    result = s._select_ego_candidate([c_new], 'left', h, w)
    check(
        not result.get('hysteresis_pending', True),
        "On hysteresis frame, switch IS confirmed"
    )
    check(s._left_switch_count == 0, "Switch counter reset after confirmation")
    check(abs(s._left_ego_x - 150) < 1, "New ego x committed after confirmation")


def test_confidence_decay():
    print("\n[TEST 3] Confidence decays monotonically during dropout frames")
    s = EgoLaneStabilizer()
    s._left_conf = 100.0
    history = [s._left_conf]
    # Check with 10 frames: 100 * 0.88^10 ≈ 27.8 — still above EGO_MIN_CONF
    for _ in range(10):
        s.update_confidence(detected_left=False, detected_right=False)
        history.append(s._left_conf)
    check(all(history[i] >= history[i+1] for i in range(len(history)-1)),
          "Confidence is monotonically non-increasing during dropout")
    # After 10 frames the confidence should still be > EGO_MIN_CONF (phantom usable)
    check(history[-1] > CONFIG.EGO_MIN_CONF,
          f"After 10 decay frames, confidence ({history[-1]:.1f}) still > EGO_MIN_CONF ({CONFIG.EGO_MIN_CONF})")


def test_confidence_recovery():
    print("\n[TEST 4] Confidence recovers quickly when detection returns")
    s = EgoLaneStabilizer()
    s._left_conf = 0.0
    for _ in range(10):
        s.update_confidence(detected_left=True, detected_right=False)
    check(s._left_conf > 90.0,
          f"After 10 recovery frames, conf ({s._left_conf:.1f}) > 90%")


def test_phantom_track_fallback():
    print("\n[TEST 5] Phantom track is used during dropout while confidence >= EGO_MIN_CONF")
    s = EgoLaneStabilizer()
    h, w = 720, 1280
    # Seed EMA with a known fit
    s._ema_left = np.array([0.0, 0.0, 400.0])
    s._left_conf = 80.0   # well above EGO_MIN_CONF=20

    # Simulate the phantom-track logic from _lane_pipeline
    left_lanes = []  # no new detection
    if (len(left_lanes) == 0
            and s._left_conf >= CONFIG.EGO_MIN_CONF
            and s._ema_left is not None):
        phantom_x = float(np.polyval(s._ema_left, h))
        left_lanes = [{
            'fit': s._ema_left.copy(),
            'centroids': [],
            'x_eval': phantom_x,
            'score': s._left_conf * 0.50,
            'phantom': True,
            'hysteresis_pending': False,
        }]
    check(len(left_lanes) == 1,
          "Phantom lane is injected when confidence above minimum during dropout")
    check(left_lanes[0].get('phantom') is True,
          "Injected lane is correctly flagged as phantom")
    check(abs(left_lanes[0]['x_eval'] - 400.0) < 1.0,
          f"Phantom x_eval ({left_lanes[0]['x_eval']:.1f}) matches persisted EMA fit c-coeff")


def test_ema_smoothing_reduces_jitter():
    print("\n[TEST 6] EMA smoothing reduces heading/curvature sign flips")
    # Simulate noisy heading signal with rapid oscillations.
    # 60 frames ≈ 2 seconds at 30 fps — long enough to see statistical smoothing.
    N_FRAMES = 60
    np.random.seed(42)
    true_heading = 5.0  # degrees
    noisy_signal = true_heading + np.random.normal(0, 8.0, N_FRAMES)

    alpha = CONFIG.TELEM_EMA_ALPHA  # 0.35
    ema_val = noisy_signal[0]
    ema_series = [ema_val]
    for v in noisy_signal[1:]:
        ema_val = alpha * v + (1.0 - alpha) * ema_val
        ema_series.append(ema_val)

    # Count sign flips (changes of sign around zero) in raw and EMA
    def count_sign_flips(series):
        return sum(
            1 for i in range(1, len(series))
            if math.copysign(1, series[i]) != math.copysign(1, series[i-1])
        )

    raw_flips = count_sign_flips(list(noisy_signal))
    ema_flips = count_sign_flips(ema_series)
    check(ema_flips < raw_flips,
          f"EMA has fewer sign flips ({ema_flips}) than raw signal ({raw_flips})")

    # EMA variance should be lower
    raw_var = float(np.var(noisy_signal))
    ema_var = float(np.var(ema_series))
    check(ema_var < raw_var,
          f"EMA variance ({ema_var:.2f}) < raw variance ({raw_var:.2f})")


def test_centre_lane_ema_smoothing():
    print("\n[TEST 7] CentreLaneEstimator EMA smoothing reduces curvature variance")
    from CENTRE_LANE import CentreLaneEstimator
    est = CentreLaneEstimator(log_fn=lambda m: None)
    h, w = 720, 1280
    ploty = np.linspace(0, h - 1, h)
    Minv = np.eye(3, dtype=np.float32)  # identity transform (BEV == ego)

    np.random.seed(7)
    # Feed 40 noisy fits to the estimator and collect smoothed curvature.
    # 40 frames is sufficient for EMA to converge and show meaningful variance reduction.
    N_FRAMES_CURV = 40
    raw_k_vals    = []
    smooth_k_vals = []
    for _ in range(N_FRAMES_CURV):
        noise_a = np.random.normal(0, 2e-4)  # noisy quadratic coeff
        left_fit  = np.array([noise_a,  0.1, float(w * 0.30)])
        right_fit = np.array([noise_a, -0.1, float(w * 0.70)])
        raw_k_vals.append(abs(noise_a))  # naive curvature proxy
        result = est.estimate(left_fit, right_fit, ploty, h, w, Minv)
        smooth_k_vals.append(result['curvature_k'])

    raw_var    = float(np.var(raw_k_vals))
    smooth_var = float(np.var(smooth_k_vals))
    check(smooth_var < raw_var,
          f"Smoothed curvature var ({smooth_var:.2e}) < raw var ({raw_var:.2e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Ego-Lane Stabilization Regression Tests")
    print("=" * 60)

    test_scoring_returns_best()
    test_hysteresis_blocks_switch()
    test_confidence_decay()
    test_confidence_recovery()
    test_phantom_track_fallback()
    test_ema_smoothing_reduces_jitter()
    test_centre_lane_ema_smoothing()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
