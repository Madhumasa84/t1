"""Stage 2 sitting-timer checks: one labeled clip through the real saved-video pipeline, scored against hand labels.

    python stage2_checks/timer_check.py --labels LABELS --run      run the models, save frames and results
    python stage2_checks/timer_check.py --labels LABELS --check    rescore the saved frames (no models) and compare

LABELS is a *_labels.json file in this folder, written before the clip's first run; its frames and results are
saved beside it as *_frames.json.gz and *_results.json. The pipeline is used as committed:
compare_video.compare with pipeline=True. The script only observes it, by wrapping estimate_posture, PostureHold
and gated_pose_view to record each frame's person boxes, track IDs, raw and final posture labels, and which boxes
received a pose.

Each labeled stand-up should stop that person's sitting timer, and each sit-down should start it, within
TOLERANCE seconds. Anyone else seated through that window should keep counting.
"""

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
JOINT_CONFIDENCE = .30  # posture.JOINT_CONFIDENCE at 1e5e782; the run refuses a different value
SITTING, UNCLEAR, TOLERANCE = "sitting", "unclear", 1.0
HIPS_KNEES = (11, 12, 13, 14)  # COCO joints the sitting rules need, besides the shoulders
TRANSITIONS = ("stand_up", "sit_down")


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def run(labels, videos_root, output):
    sys.path.insert(0, str(HERE.parent / "src"))
    from home_tower import compare_video as video

    if video.JOINT_CONFIDENCE != JOINT_CONFIDENCE:
        raise SystemExit("posture.JOINT_CONFIDENCE changed; this check keeps the estimator fixed")
    clip = labels["clip"]
    path = videos_root / clip["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
        raise SystemExit(f"{path} is not the labeled file (sha256 differs)")
    frames, state = [], {}
    real_estimate, real_view = video.estimate_posture, video.gated_pose_view

    def estimate(joints, box):
        state["pose_boxes"].append((list(box), [round(joints[joint][2], 2) for joint in HIPS_KNEES]))
        return real_estimate(joints, box)

    class Hold(video.PostureHold):
        def update(self, observations, frame):
            state["raw"] = dict(observations)
            return super().update(observations, frame)

    def view(image, people, pose, track_ids, *rest):
        state["pose_boxes"], state["raw"] = [], {}
        result = real_view(image, people, pose, track_ids, *rest)
        boxes = people.boxes.xyxy.tolist()
        matched = {max(range(len(boxes)), key=lambda index: iou(boxes[index], pose_box)): confidences
                   for pose_box, confidences in state["pose_boxes"]}
        rows = []
        for index, box in enumerate(boxes):
            track = track_ids.get(index)
            rows.append([round(value, 1) for value in box] + [
                track, state["raw"].get(track), result[-1].get(track), index in matched, matched.get(index)])
        frames.append({"t": round(rest[-1], 4), "width": image.shape[1], "height": image.shape[0], "people": rows})
        return result

    video.estimate_posture, video.PostureHold, video.gated_pose_view = estimate, Hold, view
    models = video.load_models((video.POSE_GATE_MODEL, video.MODELS[-1]))
    video.compare(path, output, models, pipeline=True)
    report = json.loads((output / f"{path.stem}_scores.json").read_text())
    return {"format": "per frame: t seconds; width and height of the processed frame; people [[x1, y1, x2, y2, "
                      "track id or null, raw posture, final posture, received a pose, pose confidence of left hip, "
                      "right hip, left knee, right knee or null]]",
            "fps": report["fps"], "frames": frames, "sitting_time_by_track_id": report["sitting_time_by_track_id"],
            "tracking": report["tracking"]}


def owner(row, width, people):
    """Which labeled person a box belongs to. With two people, A is on the left (two_person_labels.json)."""
    if len(people) == 1:
        return next(iter(people))
    x1, _, x2 = row[:3]
    if x1 < .27 * width and x2 > .72 * width:
        return "merged"
    return "A" if (x1 + x2) / 2 < .45 * width else "B"


def inside(t, intervals):
    return any(start <= t < end for start, end in intervals)


def status(row):
    """First failing step for a box whose person is seated."""
    track, raw, final, posed = row[4:8]
    if track is None:
        return "tracking: box without a track ID"
    if final == SITTING:
        return SITTING
    if not posed:
        return "posture: no pose matched to the box"
    if raw != SITTING:
        return f"posture: estimator said {raw}"
    return "posture: hold after a label change"


def times_of(value):
    return value if isinstance(value, list) else [value]


def score(saved, labels):
    frames, fps = saved["frames"], saved["fps"]
    times = [frame["t"] for frame in frames]
    durations = [b - a for a, b in zip(times, times[1:])] + [1 / fps]
    people = labels["people"]
    seated = {person: people[person]["seated"] for person in people}
    by_status = {person: {} for person in people}
    evidence = {person: {"frames": 0, "box_reaches_bottom_edge": 0, "best_hip": [], "best_knee": []} for person in people}
    sitting_by_owner = {person: 0.0 for person in [*people, "merged"]}
    tracks = {}
    for frame, duration in zip(frames, durations):
        rows = {person: [] for person in [*people, "merged"]}
        for row in frame["people"]:
            who = owner(row, frame["width"], people)
            rows[who].append(row)
            if row[4] is not None:
                entry = tracks.setdefault(str(row[4]), {"frames_by_person": {}, "sitting_seconds_by_person": {}})
                entry["frames_by_person"][who] = entry["frames_by_person"].get(who, 0) + 1
                if row[6] == SITTING:
                    entry["sitting_seconds_by_person"][who] = entry["sitting_seconds_by_person"].get(who, 0) + duration
                    sitting_by_owner[who] += duration
        for person in people:
            if not inside(frame["t"], seated[person]):
                continue
            for row in rows[person][:1]:
                evidence[person]["frames"] += 1
                evidence[person]["box_reaches_bottom_edge"] += row[3] >= frame["height"] - 1
                if row[8] is not None:
                    evidence[person]["best_hip"].append(max(row[8][:2]))
                    evidence[person]["best_knee"].append(max(row[8][2:]))
            if rows[person]:
                best = min((status(row) for row in rows[person]), key=lambda value: value != SITTING)
            elif rows["merged"]:
                best = "tracking: merged with the other person"
            else:
                best = "detection: no person box"
            by_status[person][best] = by_status[person].get(best, 0.0) + duration

    for track, entry in tracks.items():
        entry["person"] = max(entry["frames_by_person"], key=entry["frames_by_person"].get)
        entry["sitting_seconds_by_person"] = {who: round(value, 3) for who, value in entry["sitting_seconds_by_person"].items()}
        entry["sitting_seconds_on_another_person"] = round(sum(
            value for who, value in entry["sitting_seconds_by_person"].items() if who != entry["person"]), 3)
        recorded = saved["sitting_time_by_track_id"].get(track, {"sitting_seconds": 0.0, "unclear_seconds": 0.0, "intervals": []})
        entry |= {key: recorded[key] for key in ("sitting_seconds", "unclear_seconds")}
        entry["sitting_intervals"] = [[i["start_seconds"], i["end_seconds"]] for i in recorded["intervals"] if i["posture"] == SITTING]

    # Interval recording: rebuild every track's intervals from its per-frame final labels and compare.
    rebuilt = {}
    for index, frame in enumerate(frames):
        labels = {str(row[4]): row[6] for row in frame["people"] if row[4] is not None}
        end = times[index + 1] if index + 1 < len(times) else times[index] + 1 / fps
        for track, label in labels.items():
            if label in (SITTING, UNCLEAR):
                runs = rebuilt.setdefault(track, [])
                if runs and runs[-1][0] == label and runs[-1][2] == frame["t"]:
                    runs[-1][2] = end
                else:
                    runs.append([label, frame["t"], end])
    recorded = {track: [[i["posture"], i["start_seconds"], i["end_seconds"]] for i in entry["intervals"]]
                for track, entry in saved["sitting_time_by_track_id"].items()}
    # Saved frame times are already rounded to 0.1 ms, so allow the report's own 1 ms rounding step.
    matches = rebuilt.keys() == recorded.keys() and all(
        len(rebuilt[track]) == len(recorded[track]) and all(
            label == other and abs(start - other_start) <= .001 and abs(end - other_end) <= .001
            for (label, start, end), (other, other_start, other_end) in zip(rebuilt[track], recorded[track]))
        for track in rebuilt)

    result = {"interval_recording_matches_frame_labels": matches, "people": {}, "tracks": tracks}
    for person, label in people.items():
        expected = sum(end - start for start, end in label["seated"])
        counted = by_status[person].get(SITTING, 0.0)
        owned = [track for track, entry in tracks.items() if entry["person"] == person]
        outside = sum(duration for frame, duration in zip(frames, durations) if not inside(frame["t"], seated[person])
                      for row in frame["people"] if owner(row, frame["width"], people) == person and row[6] == SITTING)
        result["people"][person] = {
            "who": label["who"], "track_ids": owned,
            "expected_sitting_seconds": round(expected, 3),
            "sitting_seconds_counted_on_this_person": round(counted, 3),
            "sitting_seconds_on_own_tracks": round(sum(tracks[track]["sitting_seconds"] for track in owned), 3),
            "missed_seconds_by_first_failing_step": {key: round(value, 3) for key, value in sorted(by_status[person].items())
                                                     if key != SITTING},
            "sitting_seconds_while_not_seated": round(outside, 3),
            "while_seated": {
                "frames_with_a_box": evidence[person]["frames"],
                "boxes_reaching_the_bottom_edge": evidence[person]["box_reaches_bottom_edge"],
                "frames_with_a_pose": len(evidence[person]["best_hip"]),
                "frames_with_a_hip_at_or_above_cutoff": sum(value >= JOINT_CONFIDENCE for value in evidence[person]["best_hip"]),
                "frames_with_a_knee_at_or_above_cutoff": sum(value >= JOINT_CONFIDENCE for value in evidence[person]["best_knee"]),
                "median_best_hip_confidence": median(evidence[person]["best_hip"]) if evidence[person]["best_hip"] else None,
                "median_best_knee_confidence": median(evidence[person]["best_knee"]) if evidence[person]["best_knee"] else None,
            },
        }
    result["sitting_seconds_on_merged_boxes"] = round(sitting_by_owner["merged"], 3)
    result["sitting_seconds_on_a_track_of_another_person"] = round(
        sum(entry["sitting_seconds_on_another_person"] for entry in tracks.values()), 3)

    def sitting(person):
        return sorted(interval for track in result["people"][person]["track_ids"]
                      for interval in tracks[track]["sitting_intervals"])

    # A stand-up should end the person's sitting near the labeled time, and a sit-down should start it.
    result["transitions"], result["other_timers"] = [], []
    for person, label in people.items():
        for kind in TRANSITIONS:
            for labeled in times_of(label.get(kind, [])):
                window = (labeled - TOLERANCE, labeled + TOLERANCE)
                if kind == "stand_up":
                    ends = [end for start, end in sitting(person) if start < window[1]]
                    measured = max(ends) if ends else None
                else:
                    starts = [start for start, end in sitting(person) if end > window[0]]
                    measured = min(starts) if starts else None
                result["transitions"].append({
                    "person": person, "kind": kind, "labeled": labeled,
                    "timer_stops" if kind == "stand_up" else "timer_starts": measured,
                    "error_seconds": None if measured is None else round(measured - labeled, 3),
                    "within_tolerance": measured is not None and abs(measured - labeled) <= TOLERANCE,
                })
                for other, other_label in people.items():
                    if other == person or not any(start <= window[0] and end >= window[1]
                                                  for start, end in other_label["seated"]):
                        continue
                    spans = [[start, end] for start, end in sitting(other) if start < window[1] and end > window[0]]
                    result["other_timers"].append({
                        "person": other, "during": f"{person} {kind} at {labeled}", "window": list(window),
                        "sitting_intervals_overlapping_window": spans,
                        "kept_counting": any(start <= window[0] and end >= window[1] for start, end in spans),
                    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--labels", type=Path, required=True, help="a *_labels.json file in stage2_checks")
    parser.add_argument("--videos-root", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path, help="keep the annotated video and report here")
    args = parser.parse_args()
    labels_path = HERE / args.labels.name
    if not labels_path.name.endswith("_labels.json") or not labels_path.is_file():
        parser.error("--labels must name a *_labels.json file in stage2_checks")
    labels = json.loads(labels_path.read_text())
    stem = labels_path.name.removesuffix("_labels.json")
    frames_path, results_path = HERE / f"{stem}_frames.json.gz", HERE / f"{stem}_results.json"
    if args.run:
        with tempfile.TemporaryDirectory() as temporary:
            output = args.output or Path(temporary)
            output.mkdir(parents=True, exist_ok=True)
            saved = run(labels, args.videos_root, output)
        with gzip.GzipFile(frames_path, "wb", mtime=0) as target:
            target.write(json.dumps(saved, separators=(",", ":")).encode())
        result = score(saved, labels)
        results_path.write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = score(json.loads(gzip.open(frames_path, "rt").read()), labels)
    print(json.dumps(result, indent=2))
    if args.check:
        same = result == json.loads(results_path.read_text())
        print(f"{results_path.name} {'matches' if same else 'DIFFERS'}")
        sys.exit(not same)


if __name__ == "__main__":
    main()
