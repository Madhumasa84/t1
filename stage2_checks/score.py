"""Score the Stage 2 pose-confidence test against the hand labels. Standard library only.

    python stage2_checks/score.py            print the tables and the reconciliation
    python stage2_checks/score.py --check    also compare with results.json (exit 1 on any difference)
    python stage2_checks/score.py --write    rewrite results.json

Inputs, all beside this file:
    labels.json            hand-labelled intervals in seconds from clip start, written before any model run
    frame_labels.json.gz   final posture label per track and frame, for the current code and for the 0.70 rule
                           (regenerate with run_versions.py)

How time is counted
    A frame counts as sitting when any track's final label is sitting, so a frame is counted once however many
    people are in view. A frame lasts until the next frame's timestamp; the last frame lasts 1/fps.
    Lying clips:  seconds labelled sitting inside the labelled lying interval (false sitting).
    Seated clips: seconds labelled sitting inside the labelled seated intervals (sitting kept).
    Sit-down:     start of the first sitting run that reaches into the seated interval beginning at the labelled
                  sit-down (runs ending more than 1 s before it are ignored), minus the labelled time.
    Totals add clips. The two UP-Fall files are two camera views of the same 5 seconds, so they add 10 s of
    labelled seated time for one 5 s performance; the totals are also printed with that pair counted once.
"""

import gzip
import json
import sys
from itertools import groupby
from pathlib import Path

HERE = Path(__file__).parent
VERSIONS = {"current": 4, "rule_0.70": 5}  # index of the label inside a track entry
PAIRED_VIEWS = ("upfall_cam1", "upfall_cam2")


def seconds_inside(times, flags, intervals):
    return sum(max(0.0, min(times[i + 1], end) - max(times[i], start))
               for i, flag in enumerate(flags) if flag for start, end in intervals)


def frames_inside(times, flags, intervals):
    return sum(flag and any(start <= times[i] < end for start, end in intervals) for i, flag in enumerate(flags))


def sitting_runs(times, flags):
    runs = []
    for value, group in groupby(range(len(flags)), lambda i: flags[i]):
        index = list(group)
        if value:
            runs.append((times[index[0]], times[index[-1] + 1]))
    return runs


def score():
    labels = json.loads((HERE / "labels.json").read_text())
    with gzip.open(HERE / "frame_labels.json.gz", "rt") as source:
        recorded = json.load(source)["clips"]
    results = {}
    for clip, label in labels.items():
        if clip.startswith("_"):
            continue
        frames = recorded[clip]["frames"]
        times = [frame["t"] for frame in frames]
        times.append(times[-1] + 1 / recorded[clip]["fps"])
        every_frame = [True] * len(frames)
        entry = {"group": label["group"], "frames": len(frames), "video_s": round(times[-1], 2)}
        for kind in ("lying", "seated", "standing"):
            if kind in label:
                entry[f"{kind}_labelled_s"] = round(sum(end - start for start, end in label[kind]), 2)
                entry[f"{kind}_covered_s"] = round(seconds_inside(times, every_frame, label[kind]), 2)
                entry[f"{kind}_frames"] = frames_inside(times, every_frame, label[kind])
        for version, column in VERSIONS.items():
            sitting = [any(track[column] == "sitting" for track in frame["tracks"].values()) for frame in frames]
            scored = {"sitting_frames": sum(sitting)}
            if "lying" in label:
                scored["false_sitting_s"] = round(seconds_inside(times, sitting, label["lying"]), 2)
                scored["false_sitting_frames"] = frames_inside(times, sitting, label["lying"])
            if "seated" in label:
                scored["sitting_kept_s"] = round(seconds_inside(times, sitting, label["seated"]), 2)
                scored["sitting_kept_frames"] = frames_inside(times, sitting, label["seated"])
            if "body_in_frame_until" in label:
                scored["sitting_kept_while_body_in_frame_s"] = round(
                    seconds_inside(times, sitting, [(0, label["body_in_frame_until"])]), 2)
            if "standing" in label:
                scored["sitting_while_standing_s"] = round(seconds_inside(times, sitting, label["standing"]), 2)
            runs = sitting_runs(times, sitting)
            if "sit_down" in label:
                start, end = next(interval for interval in label["seated"] if interval[0] == label["sit_down"])
                reached = [run for run in runs if run[1] > start - 1 and run[0] < end]
                scored["first_sitting_label_s"] = round(reached[0][0], 2) if reached else None
                scored["sit_down_error_s"] = round(reached[0][0] - start, 2) if reached else None
            if "stand_up" in label:
                before = [run for run in runs if run[0] < label["stand_up"] + 1]
                scored["sitting_stopped_s"] = round(before[0][1], 2) if before else None
            tracks = {}
            for frame in frames:
                for track, value in frame["tracks"].items():
                    seen = tracks.setdefault(track, {"frames": 0, "sitting_frames": 0, "x": 0.0})
                    seen["frames"] += 1
                    seen["sitting_frames"] += value[column] == "sitting"
                    seen["x"] += (value[0] + value[2]) / 2
            scored["per_track"] = {track: {"frames": seen["frames"], "sitting_frames": seen["sitting_frames"],
                                           "mean_box_centre_x": round(seen["x"] / seen["frames"])}
                                   for track, seen in tracks.items()}
            entry[version] = scored
        results[clip] = entry

    totals = {}
    for name, skip in (("all 18 clip views", ()), ("UP-Fall pair counted once, camera 1", PAIRED_VIEWS[1:]),
                       ("UP-Fall pair counted once, camera 2", PAIRED_VIEWS[:1])):
        used = [entry for clip, entry in results.items() if clip not in skip]
        total = {"clips": len(used),
                 "lying_labelled_s": round(sum(e.get("lying_labelled_s", 0) for e in used), 2),
                 "seated_labelled_s": round(sum(e.get("seated_labelled_s", 0) for e in used), 2)}
        for version in VERSIONS:
            total[version] = {
                "false_sitting_s": round(sum(e[version].get("false_sitting_s", 0) for e in used), 2),
                "sitting_kept_s": round(sum(e[version].get("sitting_kept_s", 0) for e in used), 2)}
        totals[name] = total
    return {"clips": results, "totals": totals}


def report(result):
    clips, a, b = result["clips"], *VERSIONS
    print("Reconciliation with labels.json (labelled seconds, seconds covered by processed frames, frames)")
    for clip, entry in clips.items():
        parts = [f"{kind} {entry[f'{kind}_labelled_s']:.2f} s labelled, {entry[f'{kind}_covered_s']:.2f} s covered, "
                 f"{entry[f'{kind}_frames']} fr" for kind in ("lying", "seated", "standing") if f"{kind}_labelled_s" in entry]
        print(f"  {clip:13s} video {entry['video_s']:5.2f} s, {entry['frames']:3d} fr | " + " | ".join(parts))
    print("\nLying under covers: false sitting, current code -> 0.70 rule")
    for clip, entry in clips.items():
        if "lying_labelled_s" in entry:
            print(f"  {clip:13s} {entry['lying_labelled_s']:5.2f} s lying | {entry[a]['false_sitting_s']:.2f} s "
                  f"({entry[a]['false_sitting_frames']} fr) -> {entry[b]['false_sitting_s']:.2f} s "
                  f"({entry[b]['false_sitting_frames']} fr)")
    print("\nSeated: sitting kept, current code -> 0.70 rule | sit-down labelled -> first sitting label")
    for clip, entry in clips.items():
        if "seated_labelled_s" in entry:
            line = (f"  {clip:13s} {entry['seated_labelled_s']:5.2f} s seated | {entry[a]['sitting_kept_s']:.2f} s -> "
                    f"{entry[b]['sitting_kept_s']:.2f} s")
            if "first_sitting_label_s" in entry[a]:
                line += (f" | first sitting label {entry[a]['first_sitting_label_s']} (error "
                         f"{entry[a]['sit_down_error_s']}) -> {entry[b]['first_sitting_label_s']}")
            if "sitting_while_standing_s" in entry[a]:
                line += f" | sitting while standing {entry[a]['sitting_while_standing_s']} s"
            if "sitting_stopped_s" in entry[a]:
                line += f" | first sitting run ended {entry[a]['sitting_stopped_s']}"
            print(line)
    print("\nPer-track sitting frames where a clip has more than one track (current code)")
    for clip, entry in clips.items():
        if len(entry[a]["per_track"]) > 1:
            print(f"  {clip:13s} " + "; ".join(
                f"track {track} (box centre x {seen['mean_box_centre_x']}): {seen['sitting_frames']} of {seen['frames']} fr"
                for track, seen in entry[a]["per_track"].items()))
    print("\nTotals")
    for name, total in result["totals"].items():
        print(f"  {name} ({total['clips']} clips): false sitting {total[a]['false_sitting_s']:.2f} -> "
              f"{total[b]['false_sitting_s']:.2f} of {total['lying_labelled_s']:.2f} s lying | sitting kept "
              f"{total[a]['sitting_kept_s']:.2f} -> {total[b]['sitting_kept_s']:.2f} of {total['seated_labelled_s']:.2f} s seated")


if __name__ == "__main__":
    result = score()
    report(result)
    if "--write" in sys.argv:
        (HERE / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    if "--check" in sys.argv:
        same = json.loads((HERE / "results.json").read_text()) == json.loads(json.dumps(result))
        print("\nresults.json", "matches" if same else "DIFFERS")
        sys.exit(not same)
