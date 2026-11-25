# Home Tower

Home Tower uses a fixed camera to report whether anyone is visible in a room. It runs on the CPU, shows an annotated webcam preview in the browser, and records changes in room presence in SQLite. Saved videos can also be replayed through the same detector. Sitting-time measurement is under evaluation.

## Set up

From the checkout on Linux, install FFmpeg with H.264 encoding support and set up the tested Python 3.10 environment with [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --torch-backend cpu ultralytics==8.4.216 scipy==1.15.3 lap==0.5.14 pytest
mkdir -p ../weights
for model in yolo26s yolo11n-pose; do
  curl --fail --location --output "../weights/$model.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/$model.pt"
done
```

## Run

Start the local preview, open `http://localhost:8765` in your browser, and allow camera access. Room history is saved in `room-presence.sqlite3` at the checkout root.

```bash
python src/home_tower/camera_server.py
```

To process a recording, supply its path. The annotated MP4 and JSON report are written to `pipeline_results/`.

```bash
python src/home_tower/compare_video.py --pipeline videos/example.mp4
```

## Verify

```bash
python -m pytest -q
python stage2_checks/score.py --check
```

## Repository layout

- `src/home_tower/` contains the camera server, browser page, video pipeline, and posture and presence rules.
- `tests/` contains the automated regression tests.
- `tools/` contains the photo detection and COCO evaluation scripts.
- `stage2_checks/` contains the labeled evaluation data and scoring scripts. [Evaluation guide](stage2_checks/README.md)
- `PLAN.md` tracks accepted stages and open work. [Project plan](PLAN.md)
