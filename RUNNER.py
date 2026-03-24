"""
RUNNER.py - LKA Controller (Zero-Calib / Auto-Ego Selection)
============================================================
"""

import os
import math
import time
import threading
import queue
import cv2
import numpy as np
import socket
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk

# --- FORCE WINDOWS HIGH-DPI AWARENESS ---
if os.name == 'nt':
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import CONFIG
from LANE import YOLOPInference
from LOGGING import UnifiedLogger
from CONTROL import StanleyController
from datetime import datetime

# --- Ultra HD Target Geometry ---
CTRL_W = 460
VID_W  = 1280
VID_H  = 720
MASK_W = 1280
MASK_H = 180
WIN_W  = CTRL_W + VID_W + 40
WIN_H  = VID_H + MASK_H + 120

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def _np_to_photoimage(frame_bgr, target_w, target_h):
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image=img)

class LKA_App:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Autonomous LKA Perception & Control Dashboard")
        
        # Maximize to screen resolution — prevents the window from being
        # clipped or spawning too small on various screen configurations.
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        self.root.minsize(1400, 900)
        try:
            self.root.state('zoomed')          # Windows / some WMs
        except tk.TclError:
            pass
        
        self.is_running = False
        self.stop_event = threading.Event()
        
        self.source_path = ctk.StringVar(value="")
        self.input_mode  = ctk.StringVar(value="camera")
        self.show_drivable_area = ctk.BooleanVar(value=False)
        
        # Hardcoded ROI from CONFIG - No more manual dragging
        self.calib_points = list(CONFIG.DEFAULT_ROI_POINTS)

        self._setup_gui()
        
        self.logger = UnifiedLogger(gui_callback=self.log)
        self.controller = StanleyController(log_fn=self.log)

        self._draw_placeholder(self.canvas_main, VID_W, VID_H, "SYSTEM READY | STANDBY")
        self._draw_placeholder(self.canvas_mask, MASK_W, MASK_H, "MASK UI DISABLED FOR SPEED")

    def _setup_gui(self):
        self.header = ctk.CTkFrame(self.root, height=70, corner_radius=0, fg_color="#10141a")
        self.header.pack(fill="x", side="top")
        
        title_font = ctk.CTkFont(family="Segoe UI", size=26, weight="bold")
        ctk.CTkLabel(self.header, text="❖ IPG CarMaker LKA Control Station", font=title_font, text_color="#58a6ff").pack(side="left", padx=30, pady=15)

        main_row = ctk.CTkFrame(self.root, fg_color="transparent")
        main_row.pack(fill="both", expand=True, padx=20, pady=15)

        left = ctk.CTkFrame(main_row, width=CTRL_W, corner_radius=15, fg_color="transparent")
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)
        
        lbl_font = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")

        src_frame = ctk.CTkFrame(left, corner_radius=15, fg_color="#161b22", border_width=1, border_color="#30363d")
        src_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(src_frame, text="1. DATA SOURCE", font=lbl_font, text_color="#8b949e").pack(anchor="w", padx=20, pady=(15, 10))
        
        ctk.CTkRadioButton(src_frame, text="CarMaker TCP Stream (Live)", variable=self.input_mode, value="camera", font=("Segoe UI", 14)).pack(anchor="w", padx=25, pady=8)
        ctk.CTkRadioButton(src_frame, text="Video File (Recording)", variable=self.input_mode, value="file", font=("Segoe UI", 14)).pack(anchor="w", padx=25, pady=8)
        
        file_row = ctk.CTkFrame(src_frame, fg_color="transparent")
        file_row.pack(fill="x", padx=20, pady=(10, 20))
        ctk.CTkButton(file_row, text="Browse", width=100, height=35, fg_color="#30363d", hover_color="#484f58", command=self.browse_file).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(file_row, textvariable=self.source_path, text_color="#8b949e", font=("Consolas", 11)).pack(side="left")

        ctrl_frame = ctk.CTkFrame(left, corner_radius=15, fg_color="#161b22", border_width=1, border_color="#30363d")
        ctrl_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(ctrl_frame, text="2. MISSION CONTROL", font=lbl_font, text_color="#8b949e").pack(anchor="w", padx=20, pady=(15, 10))
        
        btn_font = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        
        self.btn_run = ctk.CTkButton(ctrl_frame, text="▶ START AUTONOMOUS", height=60, font=btn_font, fg_color="#238636", hover_color="#2ea043", command=self.start_inference)
        self.btn_run.pack(fill="x", padx=20, pady=10)
        
        self.btn_stop = ctk.CTkButton(ctrl_frame, text="■ EMERGENCY STOP", height=60, font=btn_font, state="disabled", fg_color="#da3633", hover_color="#b62324", command=self.stop_system)
        self.btn_stop.pack(fill="x", padx=20, pady=(10, 10))

        self.switch_drivable = ctk.CTkSwitch(
            ctrl_frame, text="Toggle Drivable Area",
            variable=self.show_drivable_area,
            font=("Segoe UI", 14), onvalue=True, offvalue=False)
        self.switch_drivable.pack(anchor="w", padx=25, pady=(4, 18))

        log_frame = ctk.CTkFrame(left, corner_radius=15, fg_color="#161b22", border_width=1, border_color="#30363d")
        log_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(log_frame, text="TELEMETRY FEED", font=lbl_font, text_color="#8b949e").pack(anchor="w", padx=20, pady=(15, 5))
        
        self.log_text = ctk.CTkTextbox(log_frame, state='disabled', fg_color="#0d1117", text_color="#3fb950", font=("Consolas", 13), wrap="word", corner_radius=10, border_width=1, border_color="#30363d")
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        right = ctk.CTkFrame(main_row, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        vid_container = ctk.CTkFrame(right, corner_radius=15, fg_color="#161b22", border_width=2, border_color="#30363d")
        vid_container.pack(pady=(0, 20))
        
        self.canvas_main = tk.Canvas(vid_container, width=VID_W, height=VID_H, bg="#0d1117", bd=0, highlightthickness=0)
        self.canvas_main.pack(padx=5, pady=5)

        mask_container = ctk.CTkFrame(right, corner_radius=15, fg_color="#161b22", border_width=2, border_color="#30363d")
        mask_container.pack()
        
        self.canvas_mask = tk.Canvas(mask_container, width=MASK_W, height=MASK_H, bg="#0d1117", bd=0, highlightthickness=0)
        self.canvas_mask.pack(padx=5, pady=5)

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.jpg")])
        if f: self.source_path.set(f)

    def log(self, msg: str):
        full_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.root.after(0, lambda m=full_msg: self._write_log(m))

    def _write_log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def _draw_placeholder(self, canvas, w, h, text):
        arr = np.full((h, w, 3), (18, 22, 32), dtype=np.uint8)
        for i in range(0, w, 50):
            cv2.line(arr, (i, 0), (i, h), (25, 30, 40), 1)
        for i in range(0, h, 50):
            cv2.line(arr, (0, i), (w, i), (25, 30, 40), 1)
            
        photo = _np_to_photoimage(arr, w, h)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo 
        canvas.create_text(w//2, h//2, text=text, fill="#484f58", font=("Segoe UI", 20, "bold"))

    def start_inference(self):
        self.is_running = True
        self.stop_event.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.log("[SYSTEM] Launching Autonomous Stack...")
        threading.Thread(target=self._master_loop, daemon=True).start()

    def stop_system(self):
        self.stop_event.set()
        self.is_running = False

    def _master_loop(self):
        self.log("[SYSTEM] Booting YOLOPv2 Engine...")
        engine = YOLOPInference("yolopv2.pt", log_fn=self.log)

        # Queue to hold ONLY the absolute latest frame (fixes latency)
        frame_queue = queue.Queue(maxsize=1)
        
        if self.input_mode.get() == "camera":
            self.log("[NETWORK] Connecting to CarMaker RSDA Stream (172.27.192.1:2210)...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                sock.connect(("172.27.192.1", 2210))
                self.log("[NETWORK] Successfully connected to Live Stream!")
            except Exception as e:
                self.log(f"[ERROR] Connection failed: {e}")
                self.root.after(0, self._reset_ui)
                return
            
            # --- TCP RECEIVER THREAD ---
            def receive_frames():
                byte_buffer = b""
                PACKET_SIZE = 64 + (1280 * 720 * 3)
                sock.settimeout(1.0)
                while not self.stop_event.is_set():
                    try:
                        chunk = sock.recv(65536)
                        if not chunk: break
                        byte_buffer += chunk
                        
                        while len(byte_buffer) >= PACKET_SIZE:
                            packet = byte_buffer[:PACKET_SIZE]
                            byte_buffer = byte_buffer[PACKET_SIZE:]
                            
                            if len(byte_buffer) < PACKET_SIZE: 
                                frame_bytes = packet[64:]
                                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                                frame = arr.reshape((720, 1280, 3))
                                
                                if frame_queue.full():
                                    try: frame_queue.get_nowait()
                                    except queue.Empty: pass
                                frame_queue.put(frame)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if not self.stop_event.is_set():
                            self.log(f"[NETWORK ERROR] {e}")
                        break
            
            rx_thread = threading.Thread(target=receive_frames, daemon=True)
            rx_thread.start()
            cap = None
        else:
            cap = cv2.VideoCapture(self.source_path.get())
            sock = None
        
        while not self.stop_event.is_set():
            if sock:
                try:
                    frame = frame_queue.get(timeout=0.5) 
                except queue.Empty:
                    continue
            else:
                ret, frame = cap.read()
                if not ret: break

            try:
                img_out, telemetry, ll_mask = engine.process_frame(
                    frame, self.calib_points,
                    show_drivable=self.show_drivable_area.get())
                
                # --- VIRTUAL CENTER LINE FIX ---
                h_orig, w_orig = frame.shape[:2]
                poly_coeffs = None
                
                if engine._last_left_fit is not None and engine._last_right_fit is not None:
                    poly_coeffs = (engine._last_left_fit + engine._last_right_fit) / 2.0
                elif engine._last_left_fit is not None:
                    poly_coeffs = engine._last_left_fit.copy()
                    poly_coeffs[2] += 350 
                elif engine._last_right_fit is not None:
                    poly_coeffs = engine._last_right_fit.copy()
                    poly_coeffs[2] -= 350
                
                ctrl_data = self.controller.compute_and_send(poly_coeffs, h_orig, w_orig)
                
                self._draw_hud(img_out, telemetry, ctrl_data)
                
                self.root.after(0, lambda f=img_out.copy(): self._push_frame(f, self.canvas_main, VID_W, VID_H))
                    
            except Exception as e:
                self.log(f"[ERROR] Loop crash: {e}")
                
        if sock:
            sock.close()
        elif cap:
            cap.release()
            
        self.log("[SYSTEM] Engine stopped.")
        self.root.after(0, self._reset_ui)

    def _push_frame(self, frame_bgr, canvas, w, h):
        try:
            photo = _np_to_photoimage(frame_bgr, w, h)
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo
        except Exception: pass

    def _draw_hud(self, img, tel, ctrl):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 80), (13, 17, 23), -1)
        cv2.line(img, (0, 80), (w, 80), (48, 54, 61), 2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        cv2.putText(img, f"L-Conf: {tel.get('left_conf',0):.0f}%  R-Conf: {tel.get('right_conf',0):.0f}%", (20, 35), font, 0.75, (166, 212, 250), 2)
        
        cte = ctrl.get('cte_m', 0.0)
        he  = ctrl.get('heading_err_deg', 0.0)
        str_ang = ctrl.get('steer_rad', 0.0)
        mode = ctrl.get('mode', 'IDLE')
        
        ctrl_txt = f"[{mode}] | CTE: {cte:+.2f}m | Head Err: {he:+.1f} deg | Steer CMD: {str_ang:+.2f} rad"
        
        col = (0, 200, 255) if abs(cte) > 0.4 else (63, 185, 80)
        cv2.putText(img, ctrl_txt, (20, 65), font, 0.75, col, 2)

        # Steering wheel HUD widget (bottom-right corner)
        self._draw_steering_wheel(img, str_ang)

    @staticmethod
    def _draw_steering_wheel(img: np.ndarray, steer_rad: float) -> None:
        """Draw a dynamic steering wheel widget in the bottom-right corner."""
        h, w = img.shape[:2]
        outer_r = 58
        cx, cy  = w - outer_r - 20, h - outer_r - 20

        # Dark background circle
        cv2.circle(img, (cx, cy), outer_r + 6, (15, 15, 15), -1, cv2.LINE_AA)
        # Outer rim
        cv2.circle(img, (cx, cy), outer_r, (180, 180, 180), 3, cv2.LINE_AA)
        # Inner hub
        cv2.circle(img, (cx, cy), 8, (200, 200, 200), -1, cv2.LINE_AA)

        spoke_r = int(outer_r * 0.72)
        sa = math.sin(steer_rad)
        ca = math.cos(steer_rad)

        # Main spoke (rotates with steering angle)
        tip = (cx + int(sa * spoke_r), cy - int(ca * spoke_r))
        cv2.line(img, (cx, cy), tip, (0, 220, 255), 4, cv2.LINE_AA)

        # Cross spoke (perpendicular)
        cross_r = int(spoke_r * 0.65)
        cv2.line(img,
                 (cx - int(ca * cross_r), cy - int(sa * cross_r)),
                 (cx + int(ca * cross_r), cy + int(sa * cross_r)),
                 (140, 140, 140), 2, cv2.LINE_AA)

        # Angle label below the wheel
        deg = math.degrees(steer_rad)
        label = f"{deg:+.0f}\u00b0"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, label,
                    (cx - tw // 2, cy + outer_r + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    def _reset_ui(self):
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")

if __name__ == "__main__":
    root = ctk.CTk()
    app = LKA_App(root)
    root.mainloop()