# Track-Attack

## LKA (Lane-Keeping Assist) Perception & Control System

### Overview
Track-Attack is a real-time lane-keeping assist system that uses YOLOPv2 for lane detection and a Stanley controller for steering output. It connects to IPG CarMaker via TCP/UDP.

---

## Ego-Lane Stabilization Pipeline (v15)

### Root Cause of Oscillation
When dashed lane markings are intermittently absent the detector returns no candidate for one side. The original pipeline immediately fell back to a *synthetic projection* (offset copy of the other lane) with a hard confidence jump (100% → 40%). On the next frame when the dashed lane reappeared at a slightly different detected position, the polynomial fit jumped back, producing a frame-to-frame jitter in the centre-line polynomial that fed directly into oscillatory steering commands.

### Approach
A three-layer stabilization stack was added:

1. **Score-based ego candidate selection** — each sliding-window candidate is scored on:
   - *Proximity* to the expected lateral position for that ego side (left ≈ 0.6·mid, right ≈ 1.4·mid in BEV)
   - *Heading continuity* — slope similarity to the previous frame's confirmed polynomial
   - *Curvature plausibility* — penalises implausibly sharp quadratic bends

2. **Hysteresis gate** — a potential lane-identity switch is only confirmed after `EGO_SWITCH_HYSTERESIS` consecutive frames present a different candidate, preventing single-frame noise from triggering a reassignment.

3. **Confidence-decay phantom tracks** — when a lane side is not detected the running confidence decays smoothly (`EGO_CONF_DECAY` per frame). While confidence remains above `EGO_MIN_CONF`, the last known EMA polynomial is persisted as a *phantom track* so the centre-line estimator and steering controller see a stable input throughout the dashed segment, rather than abruptly switching to a synthetic offset.

4. **EMA telemetry smoothing** — heading_deg, curvature_k, and radius_m are additionally smoothed at both the `CentreLaneEstimator` level and the `process_frame` level with configurable `TELEM_EMA_ALPHA`.

### Per-Frame Algorithm Flow
1. Read frame from input video / TCP stream.
2. Crop upper region (`_CUT_HEIGHT_RATIO`) and letterbox to model input size.
3. Run TorchScript inference (YOLOPv2).
4. Decode lane probability mask; strip padding.
5. Apply BEV perspective warp.
6. Histogram peak clustering → sliding-window centroid tracking.
7. `numpy.polyfit` on validated centroids → quadratic polynomial per candidate.
8. **Score** each candidate (proximity + heading + curvature).
9. **Hysteresis gate** per side to confirm/reject lane-identity switches.
10. **Confidence decay / phantom track** fallback on dropout frames.
11. Synthetic projection if one side is still missing.
12. Kalman filter + EMA polynomial smoothing.
13. Centre-line polynomial averaging → curvature + heading derivation.
14. **EMA smoothing** on telemetry signals.
15. Render overlays (sliding-window nodes, confidence, score, curvature, heading, hysteresis state).

---

## Tunable Parameters (`CONFIG.py`)

### Ego-Lane Stabilization

| Parameter | Default | Description |
|---|---|---|
| `EGO_CONF_DECAY` | `0.88` | Confidence multiplier per frame during lane dropout. Range (0,1). Higher = slower decay / more memory. ≈12 frames for 100→20 at 0.88. |
| `EGO_CONF_RECOVERY` | `0.50` | EMA blend weight for confidence recovery when lane is re-detected. Higher = faster recovery. |
| `EGO_MIN_CONF` | `20.0` | Confidence floor below which the phantom track is abandoned and synthetic projection is used. |
| `EGO_SWITCH_HYSTERESIS` | `5` | Consecutive frames a new lane candidate must appear before a switch is confirmed. Increase to reduce sensitivity to brief detections of adjacent lanes. |

### Ego-Lane Candidate Scoring Weights
(Must sum to 1.0)

| Parameter | Default | Description |
|---|---|---|
| `EGO_W_PROXIMITY` | `0.50` | Weight for lateral proximity to expected ego-lane position. |
| `EGO_W_HEADING` | `0.30` | Weight for heading-angle continuity with prior frame. |
| `EGO_W_CURVATURE` | `0.20` | Weight for curvature plausibility (penalises unrealistic bends). |

### Temporal Signal Smoothing

| Parameter | Default | Description |
|---|---|---|
| `TELEM_EMA_ALPHA` | `0.35` | EMA alpha for heading_deg, curvature_k, radius_m. Lower = smoother / more lag. Range (0,1). |

---

## Oscillation Mitigation Tuning Guide

**Scenario: Too slow to react on sharp corners**
→ Increase `TELEM_EMA_ALPHA` (e.g. 0.50) and decrease `EGO_CONF_DECAY` (e.g. 0.80)

**Scenario: Still oscillating on long dashed sections**
→ Increase `EGO_SWITCH_HYSTERESIS` (e.g. 8–10) and increase `EGO_CONF_DECAY` (e.g. 0.92)

**Scenario: Wrong lane being selected (adjacent lane confusion)**
→ Adjust `EGO_W_PROXIMITY` upward (e.g. 0.65) and reduce `EGO_W_HEADING`

**Scenario: Phantom track drifts and causes steering error**
→ Decrease `EGO_CONF_DECAY` so phantom tracks expire faster, or reduce `EGO_MIN_CONF` threshold

---

## Validation

Run the deterministic regression suite (no GPU/model required):

```bash
python validate_stabilization.py
```

Checks:
- Score-based selection returns the correct candidate.
- Hysteresis blocks switches for < N frames and confirms after N frames.
- Confidence decays monotonically during dropout.
- Confidence recovers after re-detection.
- Phantom track is injected when confidence is above minimum.
- EMA smoothing reduces sign flips and variance in heading signals.
- `CentreLaneEstimator` EMA smoothing reduces curvature variance.

---

## Debug Overlays

The output video now shows (below the FPS line):

| Overlay | Description |
|---|---|
| `L-Conf: XX% R-Conf: XX%` | Running confidence for left/right ego lanes |
| `Hyst L:N R:N` | Hysteresis counters; non-zero = potential switch pending |
| `Curv:X.XXXX R:XXXm Head:X.Xdeg` | Smoothed curvature, radius, and heading angle |
| `L1 XX` / `R1 XX` | Ego lane label with score (0–100) |
| Dimmed track colour | Indicates a phantom (persisted) track |
