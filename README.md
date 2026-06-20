# MTU Smart Safety Monitoring

Real-time computer vision for an industrial floor: count people, check PPE, spot falls and fire, blur faces, and ping a rover when something goes wrong. Safety rules depend on *where* you are, not just *whether* you're wearing gear.

Built for **MTU Aero Engines** at Hackatum.

---

## The problem

A fixed camera (or rover cam) watches a live floor. Supervisors need to know:

- How many people are here (without double-counting when someone leaves and comes back)
- Whether PPE matches the zone: helmet and glasses on the work floor, relaxed rules in safe areas, nobody in restricted bays
- Falls, smoke/fire, and camera blind spots
- That workers on screen are not identifiable

We are five people and had one afternoon for hardware. No time for a full SLAM stack or floor-plan calibration on site.

---

## How we ended up here

We did not start with ArUco markers on phones. The path looked more like this:

1. **Core pipeline** consisting of multiple challenges, like tracking people (YOLO + ByteTrack), checking PPE, watching for falls and fire, bluring faces on output, dispatching alerts over MQTT to a dummy rover and creating a browser dashboard. That had to run on a laptop CPU, so we built `SMOOTH_MODE` and `FAST_MODE` (smaller models, skip heavy frames, async camera grab).

2. **Zones were the hard part**. MTU cares about *location*. Our first version used static vertical bands in `zones/*.json` (left safe · centre work · right restricted). It was simple enough, but it felt glued to the camera, so if you remount the cam the “factory” moves.

3. **ArUco was the pragmatic leap** and we wanted to *show* different zone layouts live (all yellow, red/green splits, etc.) without a floor map. Someone flashed a fiducial on a phone, we wired OpenCV ArUco detection into `zone_map.py`. Marker ID picks which wall you are facing; estimated distance picks “close” vs “far” layout. The layout sticks when the marker leaves frame so the demo does not flicker. Markers live on phones for now (hopefully we will get to print them). Not production-grade, but it tells the zone story on the exhibition floor.

4. **Hard lessons from running it live**. Smoke detection fired too often; we slowed inference intervals and added multi-frame alert verification (`alert_filter.py`) plus a **LIMITED** HUD mode when the image is blurry or noisy (critical alerts still go out; PPE spam to the robot is throttled). PPE heuristics falsely called bare hair “helmet”; we now trust the YOLO model when it says there is *no* helmet.

5. **Robot handoff without a server**. We use public MQTT (HiveMQ) + `robot_dashboard.html` opened in any browser + `robot_sim.py`. You will hopefully see the full loop: vision → dispatch → rover ETA.

6. **Demo polish**: press **`w`** to cycle zone layouts if markers are awkward; press **`g`** for a little Easter egg. Supervisor replay clips land in `data/replays/` for critical events.

That is the product: a hackathon system that is honest about its limits but hopefully **works live** and points clearly at a real MTU rollout.

---

## Evaluation map

You asked for innovation, technical depth, and clear communication. Here is where to look in this repo for each.

### Innovation & creativity (40%)

| What you care about | What we did |
|------------------------|-------------|
| **Originality** | Zone-dependent safety on a single camera feed, tied to a rover dispatch loop, not just “detect helmet yes/no” |
| **Unique approach** | **ArUco markers on phones** swap coloured zone layouts (green / yellow / red) without SLAM or a CAD floor plan |
| **Unconventional tech** | MQTT to a static HTML dashboard (no backend); Garfield sticker privacy mode; histogram Re-ID for headcount on CPU |
| **WOW factor** | Live colour bands react to a phone marker; restricted entry pings the robot dashboard; `g` drops Garfield on your face |

**Zone colours on the feed**

| Colour | Meaning |
|--------|---------|
| **Green** | Safe, no PPE required |
| **Yellow** | Work floor: helmet + glasses; vest checked but not required |
| **Red** | Restricted entry alert! |

Markers **0–3** = four walls of a demo cross layout. Distance < 2 m vs ≥ 2 m switches between two band layouts per wall. Person zone = bbox **foot point** inside the bands.

### Technical implementation (40%)

| What judges care about | What we did |
|------------------------|-------------|
| **Code quality** | Staged pipeline in `src/` (one file per concern); config centralised in `config.py`; this README documents behaviour |
| **Complexity** | Up to 4 YOLO models + MediaPipe + ByteTrack + ArUco pose distance + alert verification + MQTT |
| **Works as intended** | `main.py` (OpenCV), `dashboard.py` (Streamlit), `robot_sim.py` + `robot_dashboard.html` — see Quick start |

```
Camera / mock → Tracker → PPE + Compliance → Zone map (ArUco) → Environment
              → Privacy → Alert verify → UI + MQTT → robot dashboard / sim
```

| Stage | File |
|-------|------|
| Orchestration | `src/engine.py` |
| People + Re-ID | `src/tracker.py` |
| PPE | `src/ppe_inference.py` |
| Heuristics | `src/compliance.py` |
| Zones + ArUco | `src/zone_map.py` |
| Falls / smoke / blur | `src/environment.py` |
| Face blur / Garfield | `src/privacy.py` |
| Tattoo blur (prototype) | `src/tattoo.py` |
| Alert verification | `src/alert_filter.py` |
| MQTT dispatch | `src/dispatcher.py` |
| Marker PNG generator | `generate_marker.py` |

### Presentation & communication (20%)

| What judges care about | What we did |
|------------------------|-------------|
| **Clear pitch** | Two-minute outline below |
| **Problem → solution story** | “How we ended up here” section above |
| **Live engagement** | Run the demo; use `w` / `g` during Q&A; explain ArUco honestly as a prototype |

**Two-minute pitch**

1. **Problem** — factory zones have different PPE rules; one camera should enforce that and call a rover.
2. **Trick** — CPU vision pipeline + ArUco on phones to swap layouts + MQTT to a rover.
3. **Live** — show marker → bands change; step in red → dashboard + sim react; helmet off in yellow → alert; `g` for Garfield.
4. **Next** — calibrated floor map, printed markers, private MQTT, models on the rover.

**Likely Q&A**

- *“Is this production-safe?”* — No. Demonstrator only; not certified.
- *“Are zones real?”* — Prototype. ArUco + image bands, not MTU CAD. `zone_id` in MQTT is ready for a real map later.
- *“Why public MQTT?”* — Zero setup for judges; any laptop can open the dashboard.
- *“Why Garfield?”* — Privacy still works (blur under the sticker); memorable demo moment.

---

## What is implemented

| Component | Status |
|-----------|--------|
| Live OpenCV UI | Zone overlay, labels, hazards, replay clips |
| Streamlit dashboard | Same pipeline + tunable thresholds |
| Robot dashboard + sim | MQTT alerts and fake rover ETA |
| People + Re-ID | YOLO + ByteTrack + histogram Re-ID |
| PPE | YOLO helmet/goggles + HSV vest |
| Zones | ArUco-switched layouts; manual `w` cycle |
| Environment | Falls, smoke/fire, blur detection |
| Alert hygiene | Debounce, frame verification, LIMITED mode |
| Privacy | Output blur, tattoo prototype, Garfield toggle |
| MQTT | Fall, restricted entry, PPE, fire/smoke |

**Not implemented:** calibrated floor geometry, printed marker rig, automated test suite, certified safety sign-off. `zones/monitor.json` is a legacy static layout; the live path is `src/zone_map.py` + ArUco.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ArUco markers (phones)

```bash
python generate_marker.py   # data/aruco_marker_id_0.png … _3.png
```

Full-screen on a phone (max brightness, white border visible).

### Live demo (recommended)

```bash
python robot_sim.py          # terminal 1
python main.py --source 0    # terminal 2 — w / g / q
# open robot_dashboard.html  # browser
```

### Mock (no camera)

```bash
python main.py --mock
```

PPE and falls work; ArUco layout switching and Garfield need a real feed.

### Streamlit

```bash
streamlit run dashboard.py
```

MQTT topics: `hackatum/robot/dispatch`, `hackatum/robot/status`

---

## Config (`src/config.py`)

- `FAST_MODE` / `SMOOTH_MODE` — CPU vs quality
- `PPE_*`, `SMOKE_*`, `FALL_*` — thresholds and frame intervals
- `ALERT_DEBOUNCE_SECONDS`, `ALERT_VERIFY_*`, `RELIABILITY_*`
- `PRIVACY_CENSORSHIP_MODE`, `GARFIELD_SCALE` — blur vs Garfield (`images/garfield.png`)

---

## Privacy & limits

- Processing on device; faces censored on **output** only; session pseudonyms (`Worker-A3F91C`).
- Not certified for real safety decisions.
- ArUco distance uses a synthetic camera matrix — demo accuracy only.
- Vest = colour heuristic; smoke can false-positive — tune intervals if needed.
- Public MQTT — no sensitive payloads.
- Record a backup video if the live demo might fail.

---

*Hackatum · MTU Smart Safety Monitoring*
