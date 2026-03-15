"""
CENTRE_LANE.py  —  Centre-Line Estimation Module (Optimized)
============================================================
"""

import numpy as np
import cv2

# FORCED OFF FOR SPEED Optimization: B-Spline is mathematically heavy. 
_SCIPY_OK = False 

_XM_PER_PIX  = 3.7 / 700        
_LOG_EVERY_N = 30                

class CentreLaneEstimator:
    def __init__(self, log_fn=None, xm_per_pix: float = _XM_PER_PIX):
        self._log        = log_fn if log_fn else print
        self._xm_per_pix = xm_per_pix
        self._frame_idx  = 0
        self._prev_ego_centre = None   

    def estimate(self, left_fit: np.ndarray | None, right_fit: np.ndarray | None, ploty: np.ndarray, h: int, w: int, Minv: np.ndarray) -> dict:
        self._frame_idx += 1
        log_this = (self._frame_idx % _LOG_EVERY_N == 1)

        result = {
            'centre_fit':     None,
            'centre_pts_bev': None,
            'spline_pts_bev': None,
            'ego_centre':     None,
            'curvature_k':    0.0,
            'radius_m':       float('inf'),
            'heading_deg':    0.0,
            'turn_direction': 'STRAIGHT',
            'valid':          False,
            'spline_used':    False,
            'log_this':       log_this,
        }

        if left_fit is None and right_fit is None:
            if self._prev_ego_centre is not None:
                result['ego_centre'] = self._prev_ego_centre
            return result

        centre_fit = self._polynomial_average(left_fit, right_fit, log_this)
        result['centre_fit'] = centre_fit

        try:
            cx = np.polyval(centre_fit, ploty)
            centre_pts_bev = np.column_stack((cx, ploty))  
            result['centre_pts_bev'] = centre_pts_bev
        except Exception:
            return result

        k, R, heading_deg, direction = self._compute_curvature(centre_fit, ploty, h, log_this)
        result['curvature_k']   = k
        result['radius_m']      = R
        result['heading_deg']   = heading_deg
        result['turn_direction'] = direction

        draw_pts_bev = centre_pts_bev
        ego_pts = self._bev_to_ego(draw_pts_bev, Minv, log_this)
        result['ego_centre'] = ego_pts
        result['valid']      = ego_pts is not None

        if ego_pts is not None:
            self._prev_ego_centre = ego_pts

        return result

    def _polynomial_average(self, left_fit: np.ndarray | None, right_fit: np.ndarray | None, log_this: bool) -> np.ndarray:
        half_lane_px = 1.85 / self._xm_per_pix
        if left_fit is not None and right_fit is not None:
            return (left_fit + right_fit) / 2.0
        elif left_fit is not None:
            cf = left_fit.copy()
            cf[2] += half_lane_px
            return cf
        else:
            cf = right_fit.copy()
            cf[2] -= half_lane_px
            return cf

    def _compute_curvature(self, centre_fit: np.ndarray, ploty: np.ndarray, h: int, log_this: bool) -> tuple[float, float, float, str]:
        a, b = float(centre_fit[0]), float(centre_fit[1])
        
        # Match the 40% lookahead used in CONTROL.py
        y_eval = float(h) * 0.60 

        try:
            a_m  = a * self._xm_per_pix
            dy_m = 2.0 * a_m * y_eval + b * self._xm_per_pix
            k_m  = abs(2.0 * a_m) / (1.0 + dy_m ** 2) ** 1.5
            R_m  = (1.0 / k_m) if k_m > 1e-9 else float('inf')

            x_top    = np.polyval(centre_fit, ploty[0])
            x_bottom = np.polyval(centre_fit, ploty[-1])
            dx_total = x_bottom - x_top
            heading_deg = float(np.degrees(np.arctan2(abs(dx_total), float(h))))

            if a > 1e-7: direction = "LEFT"
            elif a < -1e-7: direction = "RIGHT"
            else: direction = "STRAIGHT"

            return k_m, R_m, heading_deg, direction
        except Exception:
            return 0.0, float('inf'), 0.0, 'STRAIGHT'

    def _bev_to_ego(self, pts_bev: np.ndarray, Minv: np.ndarray, log_this: bool) -> np.ndarray | None:
        try:
            pts = pts_bev.reshape(1, -1, 2).astype(np.float32)
            return cv2.perspectiveTransform(pts, Minv).astype(np.int32)
        except Exception:
            return None