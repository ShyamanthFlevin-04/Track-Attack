"""
LOGGING.py - Unified Logging System for LKA Perception
========================================================
Responsibility:
  Centralized logging with categorized output:
    · Inference times (per-module timing)
    · Pipeline latency (end-to-end frame processing)
    · System resources (CPU/GPU load, memory usage)
    · Perception quality metrics
    · Scenario detection
  
  Provides structured output for GUI display and file logging.

Design:
  · Receives log calls from all modules (LANE, CENTRE_LANE, etc.)
  · Tracks timing for each module independently
  · Categorizes output for better visualization
  · Thread-safe for multi-threaded inference
  · Falls back gracefully if psutil unavailable
"""

import time
import threading
from collections import defaultdict
from datetime import datetime

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


class PerformanceMetrics:
    """Track inference times for individual modules."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.module_times = defaultdict(lambda: {'total': 0.0, 'count': 0, 'last': 0.0})
        self.frame_start_time = None
        self.frame_times = []
        self.frame_count = 0
        
    def start_frame(self):
        """Mark the start of a frame processing."""
        with self.lock:
            self.frame_start_time = time.perf_counter()
    
    def end_module(self, module_name: str, elapsed_ms: float):
        """Record inference time for a module."""
        with self.lock:
            metrics = self.module_times[module_name]
            metrics['last'] = elapsed_ms
            metrics['total'] += elapsed_ms
            metrics['count'] += 1
    
    def end_frame(self):
        """Mark end of frame processing and return total latency."""
        if self.frame_start_time is None:
            return None
        with self.lock:
            elapsed = (time.perf_counter() - self.frame_start_time) * 1000
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
            self.frame_count += 1
            return elapsed
    
    def get_avg_time(self, module_name: str) -> float:
        """Get average inference time for a module (ms)."""
        with self.lock:
            metrics = self.module_times[module_name]
            if metrics['count'] == 0:
                return 0.0
            return metrics['total'] / metrics['count']
    
    def get_last_time(self, module_name: str) -> float:
        """Get last inference time for a module (ms)."""
        with self.lock:
            return self.module_times[module_name]['last']
    
    def get_avg_pipeline_latency(self) -> float:
        """Get average end-to-end frame latency (ms)."""
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)
    
    def get_module_summary(self) -> dict:
        """Get summary of all module timings."""
        with self.lock:
            return {
                name: {
                    'avg_ms': metrics['total'] / metrics['count'] if metrics['count'] > 0 else 0.0,
                    'last_ms': metrics['last'],
                    'count': metrics['count']
                }
                for name, metrics in self.module_times.items()
            }


class SystemResourceMonitor:
    """Monitor CPU, GPU, and memory usage."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.last_read = {'cpu': 0.0, 'mem': 0.0, 'gpu_mem': 0.0}
        
        if _PSUTIL_OK:
            try:
                self.process = psutil.Process()
                self.process.cpu_percent(interval=None)  # warm up
            except Exception:
                self.process = None
    
    def update(self):
        """Update CPU and memory metrics."""
        if not _PSUTIL_OK or self.process is None:
            return
        
        try:
            with self.lock:
                self.last_read['cpu'] = self.process.cpu_percent(interval=0.01)
                mem_info = self.process.memory_info()
                self.last_read['mem'] = mem_info.rss / (1024 ** 2)  # Convert to MB
        except Exception:
            pass
    
    def get_metrics(self) -> dict:
        """Get current CPU and memory usage."""
        with self.lock:
            return {
                'cpu_percent': self.last_read['cpu'],
                'memory_mb': self.last_read['mem']
            }
    
    def is_available(self) -> bool:
        """Check if psutil is available."""
        return _PSUTIL_OK and self.process is not None


class PerceptionQualityTracker:
    """Track perception quality metrics over time."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.history = {
            'left_confidence': [],
            'right_confidence': [],
            'pixel_coverage': [],
            'continuity_score': [],
            'left_consistency': [],
            'right_consistency': [],
            'heading_error': [],
        }
        self.max_history = 60  # Keep last 60 frames
    
    def add_measurement(self, metrics: dict):
        """Add perception metrics for current frame."""
        with self.lock:
            if 'left_conf' in metrics:
                self.history['left_confidence'].append(metrics['left_conf'])
            if 'right_conf' in metrics:
                self.history['right_confidence'].append(metrics['right_conf'])
            if 'mask_px' in metrics:
                self.history['pixel_coverage'].append(metrics['mask_px'])
            if 'heading' in metrics:
                self.history['heading_error'].append(metrics['heading'])
            
            # Trim history
            for key in self.history:
                if len(self.history[key]) > self.max_history:
                    self.history[key].pop(0)
    
    def get_summary(self) -> dict:
        """Get summary statistics of perception quality."""
        with self.lock:
            summary = {}
            for key, values in self.history.items():
                if values:
                    summary[key] = {
                        'current': values[-1],
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
                else:
                    summary[key] = {
                        'current': 0.0,
                        'avg': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
            return summary


class ScenarioDetector:
    """Detect driving scenarios based on perception metrics."""
    
    def __init__(self):
        self.current_scenario = "IDLE"
        self.scenario_history = []
    
    def detect(self, telemetry: dict, lane_quality: dict) -> str:
        """
        Detect current scenario based on telemetry.
        
        Scenarios:
          IDLE          — no lane detection
          STRAIGHT      — lanes detected, low curvature
          GENTLE_CURVE  — moderate lane curvature
          SHARP_CURVE   — high curvature detected
          LANE_LOSS     — one or both lanes lost
          POOR_QUALITY  — low confidence in detection
        """
        left_conf = telemetry.get('left_conf', 0)
        right_conf = telemetry.get('right_conf', 0)
        
        # Check for lane loss
        if left_conf < 20 and right_conf < 20:
            self.current_scenario = "IDLE"
        elif (left_conf < 30 or right_conf < 30) and (left_conf + right_conf) < 100:
            self.current_scenario = "LANE_LOSS"
        elif left_conf < 50 or right_conf < 50:
            self.current_scenario = "POOR_QUALITY"
        else:
            # Check curvature if available from centre lane estimator
            # This would be passed via telemetry from LANE.py
            if 'curvature' in telemetry:
                k = abs(telemetry['curvature'])
                if k < 0.001:
                    self.current_scenario = "STRAIGHT"
                elif k < 0.005:
                    self.current_scenario = "GENTLE_CURVE"
                else:
                    self.current_scenario = "SHARP_CURVE"
            else:
                self.current_scenario = "STRAIGHT"  # Default when curvature unknown
        
        self.scenario_history.append(self.current_scenario)
        if len(self.scenario_history) > 100:
            self.scenario_history.pop(0)
        
        return self.current_scenario
    
    def get_scenario_summary(self) -> dict:
        """Get scenario statistics."""
        if not self.scenario_history:
            return {}
        
        counts = defaultdict(int)
        for scenario in self.scenario_history:
            counts[scenario] += 1
        
        return {
            'current': self.current_scenario,
            'history': counts
        }


class LogFormatter:
    """Format logging output with categorization."""
    
    CATEGORIES = {
        'INFERENCE': '⏱ INFERENCE TIMING',
        'PIPELINE': '🔄 PIPELINE LATENCY',
        'RESOURCES': '💻 SYSTEM RESOURCES',
        'PERCEPTION': '👁 PERCEPTION QUALITY',
        'SCENARIO': '🛣 SCENARIO DETECTION',
        'ENGINE': '⚙ ENGINE',
        'INFO': 'ℹ INFO',
        'ERROR': '⚠ ERROR',
        'DEBUG': '🐛 DEBUG',
    }
    
    @staticmethod
    def format_inference_timing(metrics: dict) -> str:
        """Format inference timing information."""
        lines = [f"\n{LogFormatter.CATEGORIES['INFERENCE']}"]
        for module, data in metrics.items():
            lines.append(
                f"  {module:15s} | Last: {data['last_ms']:6.2f}ms | "
                f"Avg: {data['avg_ms']:6.2f}ms | Count: {data['count']}"
            )
        return "\n".join(lines)
    
    @staticmethod
    def format_pipeline_latency(latency_ms: float, frame_count: int) -> str:
        """Format pipeline latency information."""
        return (f"\n{LogFormatter.CATEGORIES['PIPELINE']}\n"
                f"  E2E Latency: {latency_ms:.1f}ms | Frame: {frame_count}")
    
    @staticmethod
    def format_resources(resources: dict) -> str:
        """Format system resource information."""
        if not resources:
            return ""
        cpu = resources.get('cpu_percent', 0.0)
        mem = resources.get('memory_mb', 0.0)
        return (f"\n{LogFormatter.CATEGORIES['RESOURCES']}\n"
                f"  CPU: {cpu:.1f}% | Memory: {mem:.0f} MB")
    
    @staticmethod
    def format_perception_quality(quality: dict) -> str:
        """Format perception quality metrics."""
        lines = [f"\n{LogFormatter.CATEGORIES['PERCEPTION']}"]
        
        if 'left_confidence' in quality:
            lc = quality['left_confidence']
            lines.append(f"  Left Lane  | Curr: {lc['current']:6.1f}% | "
                        f"Avg: {lc['avg']:6.1f}% | "
                        f"[{lc['min']:6.1f}–{lc['max']:6.1f}]%")
        
        if 'right_confidence' in quality:
            rc = quality['right_confidence']
            lines.append(f"  Right Lane | Curr: {rc['current']:6.1f}% | "
                        f"Avg: {rc['avg']:6.1f}% | "
                        f"[{rc['min']:6.1f}–{rc['max']:6.1f}]%")
        
        if 'pixel_coverage' in quality:
            pc = quality['pixel_coverage']
            lines.append(f"  Coverage   | Curr: {pc['current']:8.0f}px | "
                        f"Avg: {pc['avg']:8.0f}px")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_scenario(scenario_data: dict) -> str:
        """Format scenario detection information."""
        current = scenario_data.get('current', 'UNKNOWN')
        return (f"\n{LogFormatter.CATEGORIES['SCENARIO']}\n"
                f"  Current: {current}")


class UnifiedLogger:
    """
    Main logging interface for the LKA system.
    
    Routes all log messages to:
      · GUI log window (via callback)
      · Session log file
      · Categorized metrics tracking
    """
    
    def __init__(self, gui_callback=None, file_handle=None, file_lock=None):
        """
        Parameters
        ----------
        gui_callback : callable(str) | None
            Function to call with log messages for GUI display
        file_handle : file object | None
            Open file handle for session logging
        file_lock : threading.Lock | None
            Lock for thread-safe file access
        """
        self.gui_callback = gui_callback
        self.file_handle = file_handle
        self.file_lock = file_lock
        
        # Metrics tracking
        self.perf_metrics = PerformanceMetrics()
        self.resources = SystemResourceMonitor()
        self.perception_quality = PerceptionQualityTracker()
        self.scenario_detector = ScenarioDetector()
    
    def log(self, message: str):
        """Log a message to GUI and file."""
        if self.gui_callback:
            self.gui_callback(message)
        if self.file_handle and self.file_lock:
            try:
                with self.file_lock:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self.file_handle.write(f"[{timestamp}] {message}\n")
                    self.file_handle.flush()
            except Exception:
                pass
    
    def track_module_inference(self, module_name: str, elapsed_ms: float):
        """Track inference time for a module."""
        self.perf_metrics.end_module(module_name, elapsed_ms)
    
    def track_frame_start(self):
        """Mark start of frame processing."""
        self.perf_metrics.start_frame()
    
    def track_frame_end(self) -> float:
        """Mark end of frame processing, return total latency (ms)."""
        return self.perf_metrics.end_frame()
    
    def track_perception_quality(self, telemetry: dict):
        """Track perception quality metrics."""
        self.perception_quality.add_measurement(telemetry)
    
    def detect_scenario(self, telemetry: dict) -> str:
        """Detect current driving scenario."""
        quality = self.perception_quality.get_summary()
        return self.scenario_detector.detect(telemetry, quality)
    
    def generate_status_report(self, frame_count: int) -> str:
        """
        Generate comprehensive status report.
        
        Returns formatted string with all metrics categorized.
        """
        lines = []
        
        # Inference timing
        timing = self.perf_metrics.get_module_summary()
        if timing:
            lines.append(LogFormatter.format_inference_timing(timing))
        
        # Pipeline latency
        latency = self.perf_metrics.get_avg_pipeline_latency()
        lines.append(LogFormatter.format_pipeline_latency(latency, frame_count))
        
        # System resources
        self.resources.update()
        resources = self.resources.get_metrics()
        if resources:
            lines.append(LogFormatter.format_resources(resources))
        
        # Perception quality
        quality = self.perception_quality.get_summary()
        if quality:
            lines.append(LogFormatter.format_perception_quality(quality))
        
        # Scenario detection
        scenario = self.scenario_detector.get_scenario_summary()
        if scenario:
            lines.append(LogFormatter.format_scenario(scenario))
        
        return "\n".join(lines)
    
    def get_detailed_metrics(self) -> dict:
        """Get all metrics as structured dictionary for external use."""
        return {
            'timing': self.perf_metrics.get_module_summary(),
            'pipeline_latency': self.perf_metrics.get_avg_pipeline_latency(),
            'resources': self.resources.get_metrics(),
            'perception_quality': self.perception_quality.get_summary(),
            'scenario': self.scenario_detector.get_scenario_summary()
        }


# Singleton instance (optional, can be instantiated per-session)
_default_logger = None


def get_logger(gui_callback=None, file_handle=None, file_lock=None) -> UnifiedLogger:
    """Get or create the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = UnifiedLogger(gui_callback, file_handle, file_lock)
    return _default_logger


def reset_logger():
    """Reset the default logger (useful for testing)."""
    global _default_logger
    _default_logger = None
