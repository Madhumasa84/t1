# Stage 2 pose-confidence test

Supporting files for the "Pose-confidence rule rejected" entry under Stage 2 in `PLAN.md`. The test asked whether counting a flat thigh as sitting only when the matched pose box confidence is at least 0.70 removes false sitting for people lying under covers without losing genuine sitting. It does not. The rule is not in the product code.

## Files

| File | Contents |
|---|---|
| `labels.json` | Hand labels in seconds from clip start: lying, seated and standing intervals, sit-down and stand-up times. Written on 25 November 2026 at 15:23:48 UTC, before any model ran on these clips. |
| `clips.json` | The 18 clips: local path, source and SHA-256 of the file that was scored. |
| `frame_labels.json.gz` | Final posture label for every track on every frame, for the current code and for the 0.70 rule, with detector boxes and pose confidences. Produced at commit `ed33d4a`. |
| `score.py` | Scores `frame_labels.json.gz` against `labels.json`. Standard library only. |
| `results.json` | The scored numbers that `PLAN.md` reports. |
| `run_versions.py` | Regenerates `frame_labels.json.gz` by running both versions through the saved-video pipeline. |

## Reproduce the numbers

No models or videos are needed for this step:

```bash
python stage2_checks/score.py --check
```

It prints the reconciliation with the label file, the per-clip tables and the totals, then confirms they match `results.json`.

## Rerun the models

Set up the runtime and weights as described under Stage 0 in `PLAN.md`. The Mixkit clips belong in `videos/stage2-unseen/` and the UP-Fall excerpts in `videos/up-fall/S1-A8-T1/`. Both folders are ignored by Git and stay on this machine. If a Mixkit clip is missing, download it from the address in `clips.json`; the script refuses a file whose checksum differs.

```bash
.venv/bin/python stage2_checks/run_versions.py --check
```

Each clip reports whether a fresh run gives the same labels as the saved file. All 18 clips take about 40 minutes on CPU. Add clip names to run a few, and `--videos-root DIR` when the videos live in another checkout. The script stops if `posture.estimate_posture` no longer matches the copy it carries, so update that copy before testing a changed estimator.

## How time is counted

- A frame counts as sitting when any track's final label is sitting. It is counted once however many people are in view.
- Seconds come from the video timestamps the pipeline uses. A frame lasts until the next frame's timestamp.
- Totals add the 18 clip views: 93.25 lying seconds and 105.52 seated seconds.
- The two UP-Fall files are simultaneous camera views of one participant for the same 5 seconds. They are scored as two clips, so they add 10.00 seated seconds for one 5-second performance. `score.py` also prints the totals with the pair counted once: 100.52 seated seconds either way.
- In UP-Fall camera 2 a second seated person is in view. The participant, on the left, accounts for all 82 sitting frames. The second person was labeled sitting on 67 of those same frames and adds no time.
- Sit-down timing is the start of the first sitting run that reaches into the labeled seated interval, minus the labeled sit-down time. Runs that end more than one second before the labeled time are ignored.
