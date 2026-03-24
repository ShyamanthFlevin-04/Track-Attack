"""
LANE.py - LKA Inference Engine (v15 - Ego-Lane Stabilization & Dashed-Line Robustness)
=======================================================================================
Key additions over v14:
  • Score-based ego-lane candidate selection (proximity + heading + curvature).
  • Hysteresis gate: N consecutive frames needed before a lane-switch is confirmed.
  • Confidence-decay phantom tracks: persisted fits survive short dashed-line dropouts
    with decaying confidence, preventing immediate fallback to synthetic projection.
  • EMA smoothing for final telemetry signals (heading_deg, curvature_k, radius_m).
  • Enhanced debug overlays: score, confidence, curvature, heading, hysteresis state.
"""

import torch
import cv2
import numpy as np
import time
import CONFIG
from utils.utils import select_device, letterbox
from CENTRE_LANE import CentreLaneEstimator

# ---------------------------------------------------------------------------
# Kalman Filter for 3-coefficient polynomial [a, b, c]
# ---------------------------------------------------------------------------
class PolyKalman:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.n   = 3
        self.x   = None
        self.P   = np.eye(self.n) * 1.0
        self.Q   = np.eye(self.n) * process_noise
        self.R   = np.eye(self.n) * measurement_noise
        self.I   = np.eye(self.n)
        self.initialized = False

    def predict(self):
        if not self.initialized:
            return None
        self.P = self.P + self.Q
        return self.x.copy()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=np.float64)
        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.copy()
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K) @ self.P
        return self.x.copy()

    def reset(self):
        self.x = None
        self.P = np.eye(self.n) * 1.0
        self.initialized = False


# Radius values above this are treated as effectively infinite (straight road).
# Must match _RADIUS_INF in CENTRE_LANE.py.
_RADIUS_INF = 1e6


# ---------------------------------------------------------------------------
# Main Inference Engine
# ---------------------------------------------------------------------------
class YOLOPInference:

    # =========================================================================
    # 🛠️ CLRNET ADJUSTABLE GEOMETRY SETTINGS 🛠️
    # You can adjust these to perfectly fit the road in any video!
    # =========================================================================
    
    # 1. CUT HEIGHT (0.0 to 1.0) -> Replaces manual ROI dragging.
    # Decreasing this brings the horizon down.
    _CUT_HEIGHT_RATIO = 0.50  
    
    # 2. TOP WIDTH SQUEEZE (0.0 to 1.0)
    # The width of the road near the horizon. 
    _BEV_TOP_WIDTH    = 0.20  

    # 3. BOTTOM WIDTH SPREAD (0.0 to 1.0) -> FIXES THE "A" SHAPE
    # Set to 1.0 to span the entire bottom of the camera frame.
    _BEV_BOTTOM_WIDTH = 1.00
    
    # =========================================================================

    _NWINDOWS            = 15    
    _MARGIN              = 45    
    _MINPIX_LANE         = 15    
    _HIST_PEAK_MIN       = 15    

    _XM_PER_PIX  = 3.7 / 700   
    _MAX_JUMP_M  = 2.50         
    _EMA_ALPHA   = 0.60         

    _CALIB_FRAMES     = 5     
    _CALIB_TOLERANCE  = 0.30  
    _CALIB_HARD_MIN_M = 2.0   
    _CALIB_HARD_MAX_M = 4.8   

    _LOG_EVERY_N = 1000       

    def __init__(self, weights_path: str, log_fn=None):
        self._log = log_fn if log_fn else print
        self._frame_idx = 0
        
        self._last_fps_time = time.time()
        self._fps_ema = 30.0

        self._log("[LANE] Initialising engine...")
        self.device = select_device(CONFIG.DEVICE_TARGET)
        self.half   = self.device.type != 'cpu'
        
        try:
            self.model = torch.jit.load(weights_path, map_location=self.device)
            if self.half:
                self.model.half()
            self.model.eval()
            self.model = torch.jit.optimize_for_inference(self.model)
        except Exception as e:
            raise RuntimeError(f"[LANE] Model load failed: {e}")

        self._stream = None
        if self.device.type == 'cuda':
            self._stream = torch.cuda.Stream(device=self.device)

        self._warmup()

        self._kf_left  = PolyKalman(process_noise=5e-5, measurement_noise=5e-3)
        self._kf_right = PolyKalman(process_noise=5e-5, measurement_noise=5e-3)

        self._ema_left  = None
        self._ema_right = None

        self._M          = None
        self._Minv       = None
        self._warp_size  = None

        self._last_mask = None
        self._last_da_mask   = None   # drivable-area segmentation mask (current frame)
        self._last_left_fit  = None
        self._last_right_fit = None

        self._ploty_cache_h = None
        self._ploty         = None

        self._calib_widths       = []
        self._calib_min_lane_w_m = self._CALIB_HARD_MIN_M
        self._calib_max_lane_w_m = self._CALIB_HARD_MAX_M
        self._calib_done         = False
        self._calib_nominal_w_m  = None

        self._centre_est = CentreLaneEstimator(
            log_fn=self._log,
            xm_per_pix=self._XM_PER_PIX
        )

        # -----------------------------------------------------------------
        # Ego-lane stabilization state
        # -----------------------------------------------------------------
        # Running confidence for each side (0–100). Decays each frame when
        # no detection, recovers quickly when detection is present.
        self._left_conf  = 0.0
        self._right_conf = 0.0

        # Confirmed bottom-x of the current ego lane in BEV (used for
        # identity comparison when hysteresis-gating a potential switch).
        self._left_ego_x  = None  # float or None
        self._right_ego_x = None  # float or None

        # Hysteresis counters: how many consecutive frames has a *different*
        # lane candidate been "pending" for each side.
        self._left_switch_count  = 0
        self._right_switch_count = 0

        # EMA state for final telemetry signals (heading, curvature, radius).
        self._ema_heading  = 0.0
        self._ema_curvature = 0.0
        self._ema_radius   = float('inf')

    def process_frame(self, img_raw: np.ndarray, roi_points_norm: list,
                      show_drivable: bool = False):
        self._frame_idx += 1
        
        now = time.time()
        dt = now - self._last_fps_time
        self._last_fps_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self._fps_ema = (0.1 * current_fps) + (0.9 * self._fps_ema)

        tel = {
            'left_conf':      0.0,
            'right_conf':     0.0,
            'mask_px':        0,
            'warped_px':      0,
            'centre_valid':    False,
            'curvature_k':     0.0,
            'radius_m':        float('inf'),
            'heading_deg':     0.0,
        }

        h_orig, w_orig = img_raw.shape[:2]
        img_out = img_raw.copy()

        img_in, ratio, pad = self._preprocess(img_raw)
        ll_mask_orig, decode_dbg = self._infer_and_decode(img_in, pad, h_orig, w_orig)

        # Drivable area green carpet overlay (ADAS-style) — applied before lane lines
        if show_drivable and self._last_da_mask is not None:
            da_layer = img_out.copy()
            da_layer[self._last_da_mask > 0] = (0, 200, 0)
            cv2.addWeighted(da_layer, 0.30, img_out, 0.70, 0, img_out)

        self._refresh_warp_maps_fixed(w_orig, h_orig)

        left_lane_data = None
        right_lane_data = None
        extra_lanes_data = []

        if CONFIG.ENABLE_LANE_DETECTION and ll_mask_orig is not None:
            left_lane_data, right_lane_data, extra_lanes_data, tel = self._lane_pipeline(
                ll_mask_orig, h_orig, w_orig, tel)

        # Centre Lane logic stays on pure polynomials so control logic remains smooth
        ego_centre_poly = None
        if self._M is not None and self._Minv is not None:
            if self._ploty_cache_h != h_orig:
                self._ploty = np.linspace(0, h_orig - 1, h_orig)
                self._ploty_cache_h = h_orig
                
            centre_result = self._centre_est.estimate(
                left_fit  = self._last_left_fit,
                right_fit = self._last_right_fit,
                ploty     = self._ploty,
                h         = h_orig,
                w         = w_orig,
                Minv      = self._Minv,
            )
            ego_centre_poly = centre_result.get('ego_centre')
            tel['centre_valid']   = centre_result.get('valid', False)
            tel['curvature_k']    = centre_result.get('curvature_k', 0.0)
            tel['radius_m']       = centre_result.get('radius_m', float('inf'))
            tel['heading_deg']    = centre_result.get('heading_deg', 0.0)

        # -----------------------------------------------------------------
        # EMA smoothing for final telemetry signals
        # Reduces frame-to-frame jitter in heading, curvature, and radius.
        # Always update EMA (including when signals are 0.0 on a straight road)
        # so the filter never stays stale.
        # -----------------------------------------------------------------
        telem_alpha = float(getattr(CONFIG, 'TELEM_EMA_ALPHA', 0.35))
        self._ema_curvature = (
            telem_alpha * tel['curvature_k']
            + (1.0 - telem_alpha) * self._ema_curvature
        )
        self._ema_heading = (
            telem_alpha * tel['heading_deg']
            + (1.0 - telem_alpha) * self._ema_heading
        )
        finite_r = tel['radius_m'] if tel['radius_m'] < _RADIUS_INF else self._ema_radius
        if finite_r < _RADIUS_INF:
            self._ema_radius = (
                telem_alpha * finite_r
                + (1.0 - telem_alpha) * (self._ema_radius if self._ema_radius < _RADIUS_INF else finite_r)
            )
        # Write smoothed values back into telemetry so control/HUD gets stable numbers
        tel['curvature_k'] = self._ema_curvature
        tel['heading_deg'] = self._ema_heading
        if self._ema_radius < _RADIUS_INF:
            tel['radius_m'] = self._ema_radius

        self._last_mask = ll_mask_orig

        # ---------------------------------------------------------
        # EXACT CENTROID VISUAL RENDERING (SLIDING WINDOW SEQUENCE)
        # ---------------------------------------------------------
        align_status = "ALIGNMENT: LOST"
        align_color = (0, 0, 255)
        
        if left_lane_data is not None and right_lane_data is not None:
            if tel.get('left_conf', 0) > 40.0 and tel.get('right_conf', 0) > 40.0:
                align_status = "ALIGNMENT: DUAL LOCKED"
                align_color = (0, 255, 0)
            else:
                align_status = "ALIGNMENT: SYNTHETIC PROJECTION"
                align_color = (0, 165, 255)

        # Draw discrete points mapped exactly to the sliding windows
        def render_sliding_sequence(lane_data, color, label):
            if lane_data is None or not lane_data.get('centroids'): return
            
            # Map BEV centroids directly back to perspective
            pts = np.array(lane_data['centroids'], dtype=np.float32).reshape(1, -1, 2)
            pts_persp = cv2.perspectiveTransform(pts, self._Minv)[0]
            
            # Phantom tracks are dashed; use a dimmer colour to signal low confidence
            is_phantom = lane_data.get('phantom', False)
            draw_color = tuple(int(c * 0.55) for c in color) if is_phantom else color
            
            # Draw individual tracking nodes and connecting sequence lines
            for i in range(len(pts_persp)):
                pt = pts_persp[i]
                cv2.circle(img_out, (int(pt[0]), int(pt[1])), 6, draw_color, -1)
                
                if i > 0:
                    prev_pt = pts_persp[i-1]
                    cv2.line(img_out,
                             (int(prev_pt[0]), int(prev_pt[1])),
                             (int(pt[0]),      int(pt[1])),
                             draw_color, 3, cv2.LINE_AA)
                    
            if len(pts_persp) > 0:
                # Centroid 0 is the bottom-most sliding window (bumper)
                bottom_pt = pts_persp[0]
                score_txt = f"{lane_data.get('score', 0.0):.0f}" if 'score' in lane_data else ''
                cv2.putText(img_out, f"{label} {score_txt}",
                            (int(bottom_pt[0]) - 15, int(bottom_pt[1]) + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, draw_color, 2, cv2.LINE_AA)

        # Render Ego Lanes (L1, R1) with direct centroid mapping
        render_sliding_sequence(left_lane_data, (0, 255, 0), "L1")
        render_sliding_sequence(right_lane_data, (0, 0, 255), "R1")
        
        # Render Extra Lanes (L2, R2, etc.)
        for lane_data in extra_lanes_data:
            col = (255, 200, 0) if 'L' in lane_data['name'] else (0, 255, 255)
            render_sliding_sequence(lane_data, col, lane_data['name'])

        # --- Polynomial lane curves extrapolated to image bottom ---
        # Subtle lane-region fill (L/R polynomials must both be present)
        if (self._last_left_fit is not None and self._last_right_fit is not None
                and self._Minv is not None):
            self._draw_lane_region(img_out, self._last_left_fit,
                                   self._last_right_fit, h_orig)

        # L/R polynomial lines from y=h (bumper) to horizon — eliminates floating
        if self._last_left_fit is not None and self._Minv is not None:
            self._draw_poly_curve(img_out, self._last_left_fit,
                                  h_orig, (0, 255, 0), thickness=3)
        if self._last_right_fit is not None and self._Minv is not None:
            self._draw_poly_curve(img_out, self._last_right_fit,
                                  h_orig, (0, 100, 255), thickness=3)

        # Center line: strict midpoint, anchored at image bottom
        if (self._last_left_fit is not None and self._last_right_fit is not None
                and self._Minv is not None):
            centre_fit = (self._last_left_fit + self._last_right_fit) / 2.0
            self._draw_poly_curve(img_out, centre_fit,
                                  h_orig, (0, 200, 255), thickness=2)
        elif ego_centre_poly is not None:
            cv2.polylines(img_out, [ego_centre_poly], False, (0, 200, 255), 2, cv2.LINE_AA)

        # ---------------------------------------------------------
        # STABILIZATION DEBUG OVERLAY
        # ---------------------------------------------------------
        cv2.putText(img_out, align_status, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, align_color, 2, cv2.LINE_AA)
        cv2.putText(img_out, f"FPS: {self._fps_ema:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence per side
        lc = tel.get('left_conf',  0.0)
        rc = tel.get('right_conf', 0.0)
        cv2.putText(img_out, f"L-Conf:{lc:.0f}% R-Conf:{rc:.0f}%",
                    (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 255, 180), 2, cv2.LINE_AA)

        # Hysteresis counters (> 0 means a potential lane switch is being gated)
        lh = tel.get('left_hysteresis',  0)
        rh = tel.get('right_hysteresis', 0)
        hyst_color = (0, 200, 255) if (lh > 0 or rh > 0) else (80, 80, 80)
        cv2.putText(img_out, f"Hyst L:{lh} R:{rh}",
                    (20, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.60, hyst_color, 2, cv2.LINE_AA)

        # Curvature & heading (smoothed EMA values)
        k_disp  = tel.get('curvature_k', 0.0)
        hd_disp = tel.get('heading_deg', 0.0)
        r_disp  = tel.get('radius_m',    float('inf'))
        r_txt   = f"{r_disp:.0f}m" if r_disp < 9999 else "inf"
        cv2.putText(img_out, f"Curv:{k_disp:.4f}  R:{r_txt}  Head:{hd_disp:.1f}deg",
                    (20, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

        return img_out, tel, self._last_mask

    def _warmup(self):
        rounds = CONFIG.WARMUP_ROUNDS
        sz     = CONFIG.MODEL_STRIDE
        dtype  = torch.float16 if self.half else torch.float32
        dummy  = torch.zeros(1, 3, sz, sz, dtype=dtype, device=self.device)
        with torch.no_grad():
            for _ in range(rounds):
                self.model(dummy)

    def _preprocess(self, img0: np.ndarray):
        img, ratio, pad = letterbox(img0, new_shape=CONFIG.MODEL_STRIDE, auto=False)
        img_in = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        t = torch.from_numpy(img_in).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0
        return t.unsqueeze(0), ratio, pad

    def _infer_and_decode(self, img_in, pad, h_orig, w_orig):
        dbg = {}
        try:
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    with torch.no_grad():
                        [_, _], _seg, ll = self.model(img_in)
                self._stream.synchronize()
            else:
                with torch.no_grad():
                    [_, _], _seg, ll = self.model(img_in)

            ll_probs      = torch.sigmoid(ll)
            ll_mask_small = (ll_probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

            mh, mw = ll_mask_small.shape
            top, bottom, left, right = self._crop_bounds(pad, mh, mw)

            ll_cropped = ll_mask_small[top:bottom, left:right]
            if ll_cropped.size == 0:
                self._last_da_mask = None
                return None, dbg

            ll_full = cv2.resize(ll_cropped, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # --- Drivable-area mask (YOLOPv2 seg head) ---
            da_full = None
            try:
                if _seg is not None:
                    if _seg.dim() == 4 and _seg.shape[1] >= 2:
                        da_small = (_seg.argmax(dim=1) == 1).squeeze().cpu().numpy().astype(np.uint8)
                    else:
                        da_small = (torch.sigmoid(_seg) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                    if da_small.ndim != 2:
                        da_small = da_small.reshape(da_small.shape[-2], da_small.shape[-1])
                    da_mh, da_mw = da_small.shape
                    da_t, da_b, da_l, da_r = self._crop_bounds(pad, da_mh, da_mw)
                    da_crop = da_small[da_t:da_b, da_l:da_r]
                    if da_crop.size > 0:
                        da_full = cv2.resize(da_crop, (w_orig, h_orig),
                                             interpolation=cv2.INTER_NEAREST)
            except Exception as da_err:
                self._log(f"[LANE] DA mask decode skipped: {da_err}")
            self._last_da_mask = da_full
            return ll_full, dbg

        except Exception as e:
            self._last_da_mask = None
            return None, dbg

    def _refresh_warp_maps_fixed(self, w: int, h: int):
        if self._warp_size == (w, h):
            return

        self._warp_size  = (w, h)
        top_left_x    = w * (0.50 - (self._BEV_TOP_WIDTH / 2.0))
        top_right_x   = w * (0.50 + (self._BEV_TOP_WIDTH / 2.0))
        bottom_left_x = w * (0.50 - (self._BEV_BOTTOM_WIDTH / 2.0))
        bottom_right_x= w * (0.50 + (self._BEV_BOTTOM_WIDTH / 2.0))
        horizon_y     = h * self._CUT_HEIGHT_RATIO
        
        src_pts = np.float32([
            [top_left_x, horizon_y], 
            [top_right_x, horizon_y], 
            [bottom_right_x, h * 1.00], 
            [bottom_left_x, h * 1.00]
        ])
        dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        self._M    = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self._Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    @staticmethod
    def _crop_bounds(pad: tuple, mask_h: int, mask_w: int) -> tuple:
        """Return (top, bottom, left, right) crop indices that strip letterbox padding."""
        dw, dh = float(pad[0]), float(pad[1])
        top    = max(0, min(round(dh), mask_h - 1))
        bottom = max(top + 1, min(mask_h - round(dh), mask_h))
        left   = max(0, min(round(dw), mask_w - 1))
        right  = max(left + 1, min(mask_w - round(dw), mask_w))
        return top, bottom, left, right

    # ------------------------------------------------------------------
    # Polynomial lane-curve rendering helpers
    # ------------------------------------------------------------------
    def _draw_poly_curve(self, img_out: np.ndarray, fit: np.ndarray,
                         h: int, color: tuple, thickness: int = 3) -> None:
        """Draw a smooth polynomial curve from y=h (bumper) to the BEV horizon.

        Evaluating the fit in BEV space and back-projecting via Minv ensures
        the lane line is anchored to the physical road surface from the bottom
        of the frame up to the vanishing point — no "floating" appearance.
        """
        if fit is None or self._Minv is None:
            return
        horizon_y = int(h * self._CUT_HEIGHT_RATIO)
        w = img_out.shape[1]
        ys = np.linspace(h - 1, horizon_y, 80, dtype=np.float32)
        xs = np.polyval(fit, ys).astype(np.float32)
        valid = (xs >= 0) & (xs < w)
        if not np.any(valid):
            return
        pts_bev = np.column_stack((xs[valid], ys[valid])).reshape(1, -1, 2)
        pts_persp = cv2.perspectiveTransform(pts_bev, self._Minv)[0].astype(np.int32)
        cv2.polylines(img_out, [pts_persp], False, color, thickness, cv2.LINE_AA)

    def _draw_lane_region(self, img_out: np.ndarray, left_fit: np.ndarray,
                          right_fit: np.ndarray, h: int,
                          color: tuple = (0, 200, 80),
                          alpha: float = 0.18) -> None:
        """Alpha-blend a filled polygon between the left and right polynomial lanes."""
        if self._Minv is None:
            return
        horizon_y = int(h * self._CUT_HEIGHT_RATIO)
        w = img_out.shape[1]
        ys = np.linspace(h - 1, horizon_y, 50, dtype=np.float32)
        xs_l = np.clip(np.polyval(left_fit,  ys), 0, w - 1).astype(np.float32)
        xs_r = np.clip(np.polyval(right_fit, ys), 0, w - 1).astype(np.float32)
        # Left side going up, right side coming back down → closed polygon in BEV
        left_bev  = np.column_stack((xs_l, ys))
        right_bev = np.column_stack((xs_r, ys[::-1]))
        poly_bev  = np.vstack((left_bev, right_bev)).reshape(1, -1, 2).astype(np.float32)
        poly_persp = cv2.perspectiveTransform(poly_bev, self._Minv)[0].astype(np.int32)
        overlay = img_out.copy()
        cv2.fillPoly(overlay, [poly_persp], color)
        cv2.addWeighted(overlay, alpha, img_out, 1.0 - alpha, 0, img_out)

    # ------------------------------------------------------------------
    # Ego-lane candidate scoring
    # ------------------------------------------------------------------

    def _score_ego_candidate(self, candidate: dict, side: str, h: int, w: int) -> float:
        """
        Score a lane candidate for ego selection.  Returns 0–100.

        Three components (weights from CONFIG):
          proximity   — how close is the candidate's bottom-x to the expected
                        lateral position for this side of the ego lane?
          heading     — how similar is the candidate's polynomial slope (b-coeff)
                        to the prior-frame's confirmed ego lane?
          curvature   — plausibility check; penalise implausibly sharp bends.
        """
        fit    = candidate['fit']
        x_eval = float(candidate['x_eval'])
        mid    = w / 2.0

        # 1. Proximity score
        # Expected: left ego ≈ 0.60*mid from center, right ego ≈ 1.40*mid
        if side == 'left':
            expected_x = mid * 0.60
        else:
            expected_x = mid * 1.40
        # Normalise deviation by half-width so score = 1.0 at exact expected_x
        prox_score = max(0.0, 1.0 - abs(x_eval - expected_x) / (mid * 0.90))

        # 2. Heading continuity score (uses b-coefficient = dominant linear slope)
        prior_fit = self._last_left_fit if side == 'left' else self._last_right_fit
        if prior_fit is not None:
            db = abs(float(fit[1]) - float(prior_fit[1]))
            # db ≈ 0 → same heading; db > 0.25 → very different → score 0
            heading_score = max(0.0, 1.0 - db * 4.0)
        else:
            heading_score = 0.5  # neutral when no prior frame reference

        # 3. Curvature plausibility (a-coefficient = quadratic term)
        curv = abs(float(fit[0]))
        # Penalise curvatures larger than a gentle bend threshold
        curv_score = max(0.0, 1.0 - max(0.0, curv - 5e-4) / 5e-3)

        w_p = getattr(CONFIG, 'EGO_W_PROXIMITY',  0.50)
        w_h = getattr(CONFIG, 'EGO_W_HEADING',    0.30)
        w_c = getattr(CONFIG, 'EGO_W_CURVATURE',  0.20)

        score = (w_p * prox_score + w_h * heading_score + w_c * curv_score) * 100.0
        candidate['score'] = float(score)
        return candidate['score']

    def _select_ego_candidate(self, candidates: list, side: str, h: int, w: int):
        """
        Pick the best-scored candidate for the given ego side and apply a
        hysteresis gate to prevent rapid lane-identity switching.

        Returns the selected candidate dict (with 'score' key populated),
        or None if no candidates are available.
        """
        if not candidates:
            # No detection this frame — reset switch counter
            if side == 'left':
                self._left_switch_count = 0
            else:
                self._right_switch_count = 0
            return None

        # Score every candidate
        for c in candidates:
            self._score_ego_candidate(c, side, h, w)

        # Best by score
        best = max(candidates, key=lambda c: c['score'])

        # Retrieve the current confirmed ego-lane identity for this side
        cur_x = self._left_ego_x if side == 'left' else self._right_ego_x
        hysteresis = int(getattr(CONFIG, 'EGO_SWITCH_HYSTERESIS', 5))

        if cur_x is not None:
            # Identity threshold: within 15 % of frame width = same physical lane
            identity_thr = w * 0.15
            same_identity = abs(best['x_eval'] - cur_x) < identity_thr

            if not same_identity:
                # Potential switch — increment counter; gate until threshold reached
                if side == 'left':
                    self._left_switch_count += 1
                    pending_count = self._left_switch_count
                else:
                    self._right_switch_count += 1
                    pending_count = self._right_switch_count

                if pending_count < hysteresis:
                    # Not confirmed yet — return the candidate closest to current ego
                    stable = min(candidates, key=lambda c: abs(c['x_eval'] - cur_x))
                    stable['hysteresis_pending'] = True
                    return stable
                else:
                    # Confirmed switch — reset counter and accept new identity
                    if side == 'left':
                        self._left_switch_count = 0
                    else:
                        self._right_switch_count = 0
            else:
                # Same identity — reset switch counter
                if side == 'left':
                    self._left_switch_count = 0
                else:
                    self._right_switch_count = 0

        # Commit the selected ego x for next-frame identity comparison
        if side == 'left':
            self._left_ego_x = best['x_eval']
        else:
            self._right_ego_x = best['x_eval']

        best['hysteresis_pending'] = False
        return best

    def _lane_pipeline(self, ll_mask: np.ndarray, h: int, w: int, tel: dict):
        ll_clean = ll_mask.copy()
        horizon_y = int(h * self._CUT_HEIGHT_RATIO)
        ll_clean[:horizon_y, :] = 0
        tel['mask_px'] = int(ll_clean.sum())

        if tel['mask_px'] == 0:
            return None, None, [], tel

        binary_warped = cv2.warpPerspective(ll_clean, self._M, (w, h), flags=cv2.INTER_LINEAR)
        tel['warped_px'] = int(binary_warped.sum())

        eval_h = h // 4
        histogram = np.sum(binary_warped[eval_h:, :], axis=0, dtype=np.int32)
        mid = w / 2.0

        valid_idx = np.where(histogram > self._HIST_PEAK_MIN)[0]
        
        clusters = []
        if len(valid_idx) > 0:
            current_cluster = [valid_idx[0]]
            for i in range(1, len(valid_idx)):
                if valid_idx[i] - valid_idx[i-1] < 80: 
                    current_cluster.append(valid_idx[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [valid_idx[i]]
            clusters.append(current_cluster)

        peaks = []
        for cluster in clusters:
            cluster_hist = histogram[cluster]
            peak_x = cluster[np.argmax(cluster_hist)]
            peaks.append(peak_x)

        # -------------------------------------------------------------
        # TRUE SLIDING WINDOW IMPLEMENTATION
        # Generates exact X,Y centroids for accurate point-cloud drawing
        # -------------------------------------------------------------
        valid_lanes = []
        nonzero = binary_warped.nonzero()
        nzy = np.array(nonzero[0])
        nzx = np.array(nonzero[1])
        win_h = max(1, h // self._NWINDOWS)
        
        safe_top_y = int(h * 0.15) 

        for peak_x in peaks:
            x_cur = peak_x
            centroids = []
            
            for w_idx in range(self._NWINDOWS):
                y_lo = h - (w_idx + 1) * win_h
                y_hi = h - w_idx * win_h
                xlo = x_cur - self._MARGIN
                xhi = x_cur + self._MARGIN
                
                good_inds = np.where((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xlo) & (nzx < xhi))[0]
                
                if len(good_inds) > self._MINPIX_LANE:
                    new_x = int(np.mean(nzx[good_inds]))
                    
                    # Prevent wild hook jumps
                    if abs(new_x - x_cur) < (w * 0.10): 
                        x_cur = new_x
                    
                    # Add point to the point-cloud sequence if safe
                    if y_hi > safe_top_y:
                        centroids.append((x_cur, (y_lo + y_hi) // 2))
                        
            # Require at least 3 validated sliding window points to form a lane
            if len(centroids) >= 3:
                cent_arr = np.array(centroids)
                cent_x = cent_arr[:, 0]
                cent_y = cent_arr[:, 1]
                
                # Fit the control polynomial to the perfectly stabilized centroids
                if np.max(cent_y) - np.min(cent_y) > h * 0.15:
                    fit = np.polyfit(cent_y, cent_x, 2)
                else:
                    line_fit = np.polyfit(cent_y, cent_x, 1)
                    fit = np.array([0.0, line_fit[0], line_fit[1]])
                    
                # Store exact bottom centroid X for flawless sorting (immune to math hooks)
                bottom_eval_x = cent_x[np.argmax(cent_y)]
                
                valid_lanes.append({
                    'fit': fit,
                    'centroids': centroids,
                    'x_eval': bottom_eval_x
                })

        # ------------------------------------------------------------------
        # Score-based Ego Selection with Hysteresis Gating
        # ------------------------------------------------------------------
        # Separate raw detections by side (left of midpoint / right of midpoint)
        left_candidates  = sorted(
            [l for l in valid_lanes if l['x_eval'] < mid],
            key=lambda x: x['x_eval'], reverse=True
        )
        right_candidates = sorted(
            [l for l in valid_lanes if l['x_eval'] >= mid],
            key=lambda x: x['x_eval']
        )

        # Pick the best-scored candidate per side, applying hysteresis gate
        best_left  = self._select_ego_candidate(left_candidates,  'left',  h, w)
        best_right = self._select_ego_candidate(right_candidates, 'right', h, w)

        # ------------------------------------------------------------------
        # Confidence Update — decay on dropout, recover on fresh detection
        # ------------------------------------------------------------------
        conf_decay    = float(getattr(CONFIG, 'EGO_CONF_DECAY',    0.88))
        conf_recovery = float(getattr(CONFIG, 'EGO_CONF_RECOVERY', 0.50))
        ego_min_conf  = float(getattr(CONFIG, 'EGO_MIN_CONF',      20.0))

        if best_left is not None:
            self._left_conf = self._left_conf * (1.0 - conf_recovery) + 100.0 * conf_recovery
        else:
            self._left_conf *= conf_decay

        if best_right is not None:
            self._right_conf = self._right_conf * (1.0 - conf_recovery) + 100.0 * conf_recovery
        else:
            self._right_conf *= conf_decay

        # Build ego left/right lane lists for downstream processing
        left_lanes  = [best_left]  if best_left  is not None else []
        right_lanes = [best_right] if best_right is not None else []

        # ------------------------------------------------------------------
        # Phantom Track Fallback (dashed-line dropout handling)
        # When confidence is still above minimum, persist the last known fit
        # so control stays smooth until the lane reappears.
        # ------------------------------------------------------------------
        if len(left_lanes) == 0 and self._left_conf >= ego_min_conf and self._ema_left is not None:
            phantom_x = float(np.polyval(self._ema_left, h))
            left_lanes = [{
                'fit':               self._ema_left.copy(),
                'centroids':         [],          # no new window centroids
                'x_eval':            phantom_x,
                'score':             self._left_conf * 0.50,
                'phantom':           True,
                'hysteresis_pending': False,
            }]

        if len(right_lanes) == 0 and self._right_conf >= ego_min_conf and self._ema_right is not None:
            phantom_x = float(np.polyval(self._ema_right, h))
            right_lanes = [{
                'fit':               self._ema_right.copy(),
                'centroids':         [],
                'x_eval':            phantom_x,
                'score':             self._right_conf * 0.50,
                'phantom':           True,
                'hysteresis_pending': False,
            }]

        # ------------------------------------------------------------------
        # Synthetic Projection — if one side is still missing after phantom
        # ------------------------------------------------------------------
        lane_width_px = 3.5 / self._XM_PER_PIX
        if len(left_lanes) > 0 and len(right_lanes) == 0:
            synth = left_lanes[0].copy()
            synth['fit'] = synth['fit'].copy()
            synth['fit'][2] += lane_width_px
            synth['centroids'] = [(cx + lane_width_px, cy) for cx, cy in synth['centroids']]
            synth['phantom'] = True
            right_lanes.append(synth)
            tel['right_conf'] = min(40.0, self._right_conf)
            tel['left_conf']  = self._left_conf
        elif len(right_lanes) > 0 and len(left_lanes) == 0:
            synth = right_lanes[0].copy()
            synth['fit'] = synth['fit'].copy()
            synth['fit'][2] -= lane_width_px
            synth['centroids'] = [(cx - lane_width_px, cy) for cx, cy in synth['centroids']]
            synth['phantom'] = True
            left_lanes.append(synth)
            tel['left_conf']  = min(40.0, self._left_conf)
            tel['right_conf'] = self._right_conf
        elif len(left_lanes) > 0 and len(right_lanes) > 0:
            tel['left_conf']  = self._left_conf
            tel['right_conf'] = self._right_conf

        # Expose hysteresis state in telemetry for debug overlay
        tel['left_hysteresis']  = self._left_switch_count
        tel['right_hysteresis'] = self._right_switch_count

        # Control-Loop Kalman Smoothing
        if len(left_lanes) > 0:
            fit = left_lanes[0]['fit']
            pred = self._kf_left.predict()
            if pred is not None:
                jump_m = abs(np.polyval(fit, h) - np.polyval(pred, h)) * self._XM_PER_PIX
                if jump_m > self._MAX_JUMP_M:
                    fit = pred
                    tel['left_conf'] = 50.0
                else:
                    fit = self._kf_left.update(fit)
            else:
                fit = self._kf_left.update(fit)
            
            self._ema_left = fit if self._ema_left is None else self._EMA_ALPHA * fit + (1 - self._EMA_ALPHA) * self._ema_left
            left_lanes[0]['fit'] = self._ema_left

        if len(right_lanes) > 0:
            fit = right_lanes[0]['fit']
            pred = self._kf_right.predict()
            if pred is not None:
                jump_m = abs(np.polyval(fit, h) - np.polyval(pred, h)) * self._XM_PER_PIX
                if jump_m > self._MAX_JUMP_M:
                    fit = pred
                    tel['right_conf'] = 50.0
                else:
                    fit = self._kf_right.update(fit)
            else:
                fit = self._kf_right.update(fit)
                
            self._ema_right = fit if self._ema_right is None else self._EMA_ALPHA * fit + (1 - self._EMA_ALPHA) * self._ema_right
            right_lanes[0]['fit'] = self._ema_right

        self._last_left_fit  = left_lanes[0]['fit'] if len(left_lanes) > 0 else None
        self._last_right_fit = right_lanes[0]['fit'] if len(right_lanes) > 0 else None

        ego_left_data = left_lanes[0] if len(left_lanes) > 0 else None
        ego_right_data = right_lanes[0] if len(right_lanes) > 0 else None
        
        # Extra lanes: remaining candidates beyond the selected ego lane
        extra_lanes_data = []
        left_ego_fit_id  = id(ego_left_data['fit'])  if ego_left_data  else None
        right_ego_fit_id = id(ego_right_data['fit']) if ego_right_data else None

        extra_left_idx = 1
        for c in left_candidates:
            if id(c['fit']) == left_ego_fit_id:
                continue
            if extra_left_idx > 2:
                break
            c['name'] = f'L{extra_left_idx + 1}'
            extra_lanes_data.append(c)
            extra_left_idx += 1

        extra_right_idx = 1
        for c in right_candidates:
            if id(c['fit']) == right_ego_fit_id:
                continue
            if extra_right_idx > 2:
                break
            c['name'] = f'R{extra_right_idx + 1}'
            extra_lanes_data.append(c)
            extra_right_idx += 1

        return ego_left_data, ego_right_data, extra_lanes_data, tel