# MTU Smart Safety Monitoring System

A real-time, multi-stage Computer Vision pipeline for warehouse and industrial safety: person tracking, PPE compliance, zone-aware alerts, fall/smoke detection, privacy blur, and MQTT robot dispatch.

## Features

- **Person Detection & Re-ID**: YOLO + ByteTrack with CNN Re-ID
- **PPE Compliance**: Custom YOLO model + heuristic fallback
- **Zone-aware compliance**: Full-frame zones with per-area PPE rules and `zone_id` on MQTT dispatch
- **Fall / smoke / fire**: Pose-based falls and environmental YOLO
- **Privacy**: Face blur on display output
- **Mock mode**: Synthetic warehouse scene for demos without a camera
- **Robot dashboard**: `robot_dashboard.html` subscribes to HiveMQ dispatch topic

## Pipeline stages

1. **Tracker** (`tracker.py`) — people + persistent IDs  
2. **PPE** (`ppe_inference.py`) — helmets & glasses  
3. **Compliance** (`compliance.py`) — heuristic fallback  
4. **Zones** (`zone_map.py`) — safe / work / restricted rules  
5. **Environment** (`environment.py`) — blur, smoke, fire, falls  
6. **Privacy** (`privacy.py`) — anonymized output frame  

## Run

```bash
pip install -r requirements.txt

# Live camera
python main.py --source 0

# Mock demo (no camera)
python main.py --mock

# Streamlit dashboard
streamlit run dashboard.py
```

### Zone layouts

Zones cover the **full camera frame** (side-mounted camera). Default: **left = safe**, **center = work**, **right = restricted**.

| Key | Action |
|-----|--------|
| `w` | Cycle layout: 3 zones → full safe → full work → full restricted |
| `q` | Quit |

Configs live in `zones/monitor.json` and `zones/rover.json`. Override with `--zones-file` or `--camera-profile rover`.

## Privacy & security

- Processing runs **on device**; no cloud video upload by default  
- Faces are **blurred** before display  
- UI shows session labels like `Worker-A3F91C`, not employee IDs  
- `.gitignore` excludes camera dumps and `.env` files  

## Robot dispatch

Set `DISPATCH_BACKEND = "mqtt"` in `src/config.py`. Open `robot_dashboard.html` on any machine to receive alerts (`FALL_DETECTED`, `RESTRICTED_ENTRY`, `NO_HELMET`, etc.) with real `zone_id`.
