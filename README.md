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

## Read sitting time

Sitting time is still under evaluation. It turns the posture label shown on each tracked person into time intervals. Known errors in those labels pass straight into the totals: people lying under a duvet can be counted as sitting, and seated people behind a table or facing the camera can be missed. See Stage 2 in [PLAN.md](PLAN.md).

How intervals are counted:

- Time counts as sitting only while a track's displayed label is `sitting`. Each interval starts at the first frame with that label and ends at the first frame where the label changes or the track is missing.
- Time labeled `unclear` is reported separately and never counts as sitting. This includes the 0.3-second hold after any posture change.
- A gap of more than 3 seconds between frames ends every interval at the last frame before the gap. The gap itself adds no time.
- A track ID belongs to the tracker, not to a person. One person can receive a new ID after an occlusion, and two people tracked as one box share one ID. Don't add up totals across IDs to get a total for a person.

**Saved videos.** In `pipeline_results/NAME_scores.json`, `sitting_time_by_track_id` lists each track ID with `sitting_seconds`, `unclear_seconds`, and its `intervals`. Each interval has a `posture`, a `start_seconds`, and an `end_seconds`, measured from the recording's timestamps. Each total is the sum of that track's intervals. The last frame of the recording lasts one frame period. Track IDs match the `tracking` section of the same report, and the annotated video shows each track's running total as `sat M:SS`.

**Browser preview.** Each person's label shows the sitting total for that track, for example `ID 1 - sitting | sat 2:05`. The total includes an interval still in progress and uses elapsed time. Stopping the camera, starting a new browser session, or going more than 3 seconds without a processed frame ends coverage. Open intervals close at the last processed frame, and later frames start a new tracker run, so IDs and totals start again from zero. When the preview reuses an ID number for a new track, that track's total also starts at zero.

Each closed interval is saved in the `posture_intervals` table of `room-presence.sqlite3`, next to the room presence history. Track IDs are unique only within one `tracker_run`. This command prints the totals for each track:

```bash
python -c "import sqlite3; [print(*row) for row in sqlite3.connect('room-presence.sqlite3').execute(
  'SELECT tracker_run, track_id, posture, ROUND(SUM(seconds), 1), MIN(started_at_utc) FROM posture_intervals '
  'GROUP BY tracker_run, track_id, posture ORDER BY 5')]"
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
