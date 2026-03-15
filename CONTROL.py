"""
CONTROL.py - Dynamic Lookahead Stanley (Curve Optimized)
========================================================
"""
import math
import socket
import CONFIG

class StanleyController:
    def __init__(self, log_fn=None):
        self._log = log_fn if log_fn else print
        
        # --- TUNED FOR SMOOTHNESS & STABILITY ---
        self.k_gain = CONFIG.STANLEY_K       
        self.v_mps  = CONFIG.STANLEY_V_MPS   
        self.max_steer = CONFIG.MAX_STEER_RAD 
        self.xm_per_pix = 3.7 / 700 
        
        self.udp_ip = CONFIG.UDP_IP
        self.udp_port = CONFIG.UDP_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.frames = 0
        self.mode = "STANLEY"
        
        # --- EMA FILTER STATE ---
        self.prev_steer_cmd = 0.0
        self.ema_alpha = 0.45  # Slightly increased to react faster in curves
        
        self._log("[CONTROL] DYNAMIC LOOKAHEAD MODE ACTIVE.")

    def compute_and_send(self, centre_fit, h: int, w: int) -> dict:
        self.frames += 1
        tel = {'cte_m': 0.0, 'heading_err_deg': 0.0, 'steer_rad': 0.0, 'mode': self.mode}

        if centre_fit is None:
            # Gradually return to center if lane is lost
            smooth_cmd = self.prev_steer_cmd * 0.8 
            self._send_udp(smooth_cmd)
            self.prev_steer_cmd = smooth_cmd
            return tel

        a, b, c = centre_fit[0], centre_fit[1], centre_fit[2]
        
        # --- DYNAMIC LOOKAHEAD FOR SHARP CURVES ---
        # abs(a) is a proxy for how sharp the curve is.
        # Straight road = look far ahead (40% down from top).
        # Sharp curve = pull lookahead closer to bumper (up to 80% down).
        curve_severity = min(1.0, abs(a) * 2000) 
        lookahead_factor = 0.40 + (0.40 * curve_severity) 
        lookahead_y = float(h) * lookahead_factor
        
        # 1. CROSS-TRACK ERROR (e) -> STRICTLY ENFORCING TRUE CENTER
        car_x = w / 2.0
        lane_x = (a * (lookahead_y ** 2)) + (b * lookahead_y) + c
        cte_m = (lane_x - car_x) * self.xm_per_pix
        
        # 2. HEADING ERROR (ψ_e)
        dx_dy = (2.0 * a * lookahead_y) + b
        heading_term = math.atan(dx_dy) 
        
        # 3. CROSS-TRACK TERM
        cte_term = -math.atan((self.k_gain * cte_m) / (self.v_mps + 0.1))

        # 4. RAW STEERING COMMAND
        raw_steer_cmd = heading_term + cte_term
        
        # 5. EMA LOW-PASS FILTER (Anti-Oscillation)
        smooth_cmd = (self.ema_alpha * raw_steer_cmd) + ((1.0 - self.ema_alpha) * self.prev_steer_cmd)
        
        # Clamp command
        smooth_cmd = max(-self.max_steer, min(self.max_steer, smooth_cmd))
        self.prev_steer_cmd = smooth_cmd
        
        if self.frames % 30 == 0:
            self._log(f"Lookahead: {lookahead_factor:.2f} | CTE: {cte_m:.2f}m | CMD: {smooth_cmd:.2f} rad")
            
        self._send_udp(smooth_cmd)
        
        tel['cte_m'] = cte_m
        tel['heading_err_deg'] = math.degrees(heading_term)
        tel['steer_rad'] = smooth_cmd
        tel['mode'] = f"STANLEY (Look: {lookahead_factor:.2f})"
        
        return tel

    def _send_udp(self, steer_angle: float):
        try:
            self.sock.sendto(f"{steer_angle:.6f}".encode(), (self.udp_ip, self.udp_port))
        except Exception:
            pass