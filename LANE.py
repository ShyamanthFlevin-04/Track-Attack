"""
LANE.py - LKA Inference Engine (v14 - True Sliding-Window Sequence & Point Validation)
======================================================================================
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

    def process_frame(self, img_raw: np.ndarray, roi_points_norm: list):
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
            
            # Draw individual tracking nodes and connecting sequence lines
            for i in range(len(pts_persp)):
                pt = pts_persp[i]
                cv2.circle(img_out, (int(pt[0]), int(pt[1])), 6, color, -1)
                
                if i > 0:
                    prev_pt = pts_persp[i-1]
                    cv2.line(img_out, (int(prev_pt[0]), int(prev_pt[1])), (int(pt[0]), int(pt[1])), color, 3, cv2.LINE_AA)
                    
            if len(pts_persp) > 0:
                # Centroid 0 is the bottom-most sliding window (bumper)
                bottom_pt = pts_persp[0]
                cv2.putText(img_out, label, (int(bottom_pt[0]) - 15, int(bottom_pt[1]) + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Render Ego Lanes (L1, R1) with direct centroid mapping
        render_sliding_sequence(left_lane_data, (0, 255, 0), "L1")
        render_sliding_sequence(right_lane_data, (0, 0, 255), "R1")
        
        # Render Extra Lanes (L2, R2, etc.)
        for lane_data in extra_lanes_data:
            col = (255, 200, 0) if 'L' in lane_data['name'] else (0, 255, 255)
            render_sliding_sequence(lane_data, col, lane_data['name'])
            
        # Draw Smooth Center Line for visual reference
        if ego_centre_poly is not None:
            cv2.polylines(img_out, [ego_centre_poly], False, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.putText(img_out, align_status, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, align_color, 2, cv2.LINE_AA)
        cv2.putText(img_out, f"FPS: {self._fps_ema:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
            dw, dh = float(pad[0]), float(pad[1])
            top    = round(dh);         bottom = mh - round(dh)
            left   = round(dw);         right  = mw - round(dw)
            top    = max(0, min(top,    mh - 1))
            bottom = max(top + 1, min(bottom, mh))
            left   = max(0, min(left,   mw - 1))
            right  = max(left + 1, min(right,  mw))

            ll_cropped = ll_mask_small[top:bottom, left:right]
            if ll_cropped.size == 0:
                return None, dbg

            ll_full = cv2.resize(ll_cropped, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            return ll_full, dbg

        except Exception as e:
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

        # Independent Ego Sorting using true bottom points
        left_lanes = sorted([l for l in valid_lanes if l['x_eval'] < mid], key=lambda x: x['x_eval'], reverse=True)
        right_lanes = sorted([l for l in valid_lanes if l['x_eval'] >= mid], key=lambda x: x['x_eval'])

        # Synthetic Projection
        lane_width_px = 3.5 / self._XM_PER_PIX
        if len(left_lanes) > 0 and len(right_lanes) == 0:
            synth = left_lanes[0].copy()
            synth['fit'] = synth['fit'].copy()
            synth['fit'][2] += lane_width_px
            # Shift point cloud for drawing the synthetic lane
            synth['centroids'] = [(cx + lane_width_px, cy) for cx, cy in synth['centroids']]
            right_lanes.append(synth)
            tel['right_conf'] = 40.0
            tel['left_conf'] = 100.0
        elif len(right_lanes) > 0 and len(left_lanes) == 0:
            synth = right_lanes[0].copy()
            synth['fit'] = synth['fit'].copy()
            synth['fit'][2] -= lane_width_px
            synth['centroids'] = [(cx - lane_width_px, cy) for cx, cy in synth['centroids']]
            left_lanes.append(synth)
            tel['left_conf'] = 40.0
            tel['right_conf'] = 100.0
        elif len(left_lanes) > 0 and len(right_lanes) > 0:
            tel['left_conf'] = 100.0
            tel['right_conf'] = 100.0

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
        
        extra_lanes_data = []
        if len(left_lanes) > 1:
            left_lanes[1]['name'] = 'L2'
            extra_lanes_data.append(left_lanes[1])
        if len(left_lanes) > 2:
            left_lanes[2]['name'] = 'L3'
            extra_lanes_data.append(left_lanes[2])
        if len(right_lanes) > 1:
            right_lanes[1]['name'] = 'R2'
            extra_lanes_data.append(right_lanes[1])
        if len(right_lanes) > 2:
            right_lanes[2]['name'] = 'R3'
            extra_lanes_data.append(right_lanes[2])

        return ego_left_data, ego_right_data, extra_lanes_data, tel