"""Regenerate frame_labels.json.gz for the Stage 2 pose-confidence test. Test helper only.

    python stage2_checks/run_versions.py --check                compare a fresh run with the saved file
    python stage2_checks/run_versions.py --check upfall_cam1    only the named clips
    python stage2_checks/run_versions.py --write                replace the saved file (all clips)

Each clip in clips.json goes through this checkout's saved-video pipeline (compare_video.compare with
pipeline=True) twice: once as the code is, and once with the rule under test, where a flat thigh counts as sitting
only when the matched pose box confidence is at least GATE. The rule lives in this file. Nothing in the pipeline is
edited; estimate_posture is swapped at run time, and every call checks that the copy below still equals the real
estimator when the rule is off.

Needs the runtime and weights from PLAN.md Stage 0 (weights in the sibling ../weights directory). Clips are read
from --videos-root (default: this checkout) at the paths in clips.json. Annotated videos go to a temporary folder.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from math import hypot
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compare_video as video  # noqa: E402

GATE = .70
JOINT_CONFIDENCE, UNCLEAR = video.JOINT_CONFIDENCE, video.UNCLEAR


def estimate_with_rule(joints, box, flat_thigh_counts):
    """Copy of posture.estimate_posture at ed33d4a with one switch on the flat-thigh rule."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    sides = []
    for shoulder, hip, knee, ankle in ((5, 11, 13, 15), (6, 12, 14, 16)):
        if min(joints[shoulder][2], joints[hip][2]) < JOINT_CONFIDENCE:
            continue
        sx, sy = joints[shoulder][:2]
        hx, hy = joints[hip][:2]
        torso = hypot(hx - sx, hy - sy)
        if torso >= .12 * max(width, height) and torso > 0:
            sides.append((hx, hy, torso, (hy - sy) / torso, knee, ankle))
    if not sides:
        return UNCLEAR
    if width > 1.25 * height and all(abs(vertical) <= .5 for _, _, _, vertical, _, _ in sides):
        return "lying down"
    sitting = standing = False
    shortened_thighs = 0
    for hx, hy, torso, vertical, knee, ankle in sides:
        if vertical < .7 or joints[knee][2] < JOINT_CONFIDENCE:
            continue
        kx, ky = joints[knee][:2]
        thigh = hypot(kx - hx, ky - hy)
        if thigh < .25 * torso:
            continue
        sitting |= flat_thigh_counts and abs(ky - hy) <= .60 * thigh
        if joints[ankle][2] >= JOINT_CONFIDENCE:
            _, ay = joints[ankle][:2]
            if ky - hy >= .75 * thigh and ay - ky >= .25 * torso:
                if ky - hy <= .70 * torso and ky - hy <= .85 * (ay - ky):
                    shortened_thighs += 1
                else:
                    standing = True
    sitting |= shortened_thighs == 2
    if sitting != standing:
        return "sitting" if sitting else "standing"
    return UNCLEAR


state = {"gate": None, "rows": [], "time": None, "calls": [], "pose_confidence": {}}


class Presence(video.RoomPresence):
    def observe(self, people, now):  # the pipeline passes each frame's video timestamp here
        state["time"], state["calls"] = now, []
        return super().observe(people, now)


real_estimate = video.estimate_posture


def estimate(joints, box):
    label = real_estimate(joints, box)
    if estimate_with_rule(joints, box, True) != label:
        raise SystemExit("posture.estimate_posture has changed; update the copy in run_versions.py before testing")
    confidence = state["pose_confidence"][tuple(box)]
    if state["gate"] is not None:
        label = estimate_with_rule(joints, box, confidence >= state["gate"])
    state["calls"].append([round(confidence, 4), label])
    return label


real_view = video.gated_pose_view


def view(image, people, pose, track_ids, *rest):
    state["pose_confidence"] = dict(zip(map(tuple, pose.boxes.xyxy.tolist()), pose.boxes.conf.tolist()))
    result = real_view(image, people, pose, track_ids, *rest)
    boxes = people.boxes.xyxy.tolist()
    state["rows"].append({
        "t": round(state["time"], 4),
        "tracks": {str(track): [round(value) for value in boxes[index]] + [result[-1][track]]
                   for index, track in track_ids.items()},
        "calls": state["calls"],
    })
    return result


video.RoomPresence, video.estimate_posture, video.gated_pose_view = Presence, estimate, view


def run(path, models, gate, output):
    state["gate"], state["rows"] = gate, []
    video.compare(path, output, models, pipeline=True)
    return state["rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--videos-root", type=Path, default=HERE.parent)
    parser.add_argument("clips", nargs="*")
    args = parser.parse_args()
    manifest = json.loads((HERE / "clips.json").read_text())
    names = args.clips or list(manifest)
    if args.write and args.clips:
        parser.error("--write regenerates every clip")
    saved_path = HERE / "frame_labels.json.gz"
    saved = json.loads(gzip.open(saved_path, "rt").read()) if saved_path.exists() else None
    models = video.load_models((video.POSE_GATE_MODEL, video.MODELS[-1]))
    fresh, different = {}, []
    with tempfile.TemporaryDirectory() as output:
        for name in names:
            path = args.videos_root / manifest[name]["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
                raise SystemExit(f"{path} is not the file that was scored (sha256 differs from clips.json)")
            current = run(path, models, None, Path(output))
            fps = json.loads((Path(output) / f"{path.stem}_scores.json").read_text())["fps"]
            ruled = run(path, models, GATE, Path(output))
            frames = []
            for a, b in zip(current, ruled):
                assert a["t"] == b["t"] and list(a["tracks"]) == list(b["tracks"])
                frames.append({"t": a["t"],
                               "tracks": {track: entry + [b["tracks"][track][-1]] for track, entry in a["tracks"].items()},
                               "calls": [x + [y[1]] for x, y in zip(a["calls"], b["calls"])]})
            fresh[name] = {"fps": fps, "frames": frames}
            if args.check:
                same = saved is not None and saved["clips"].get(name) == json.loads(json.dumps(fresh[name]))
                different += [] if same else [name]
                print(f"{name}: {len(frames)} frames, {'identical to' if same else 'DIFFERENT from'} the saved labels",
                      flush=True)
    if args.write:
        commit = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or "unknown"
        document = {"pipeline_commit": commit, "rule": f"flat thigh counts as sitting only when matched pose "
                    f"box confidence >= {GATE}", "format": "per frame: t seconds; tracks {track id: [x1, y1, x2, y2, "
                    "label current code, label with the rule]}; calls [[pose box confidence, raw label current code, "
                    "raw label with the rule]]", "clips": fresh}
        with gzip.GzipFile(saved_path, "wb", mtime=0) as target:
            target.write(json.dumps(document, separators=(",", ":")).encode())
        print(f"wrote {saved_path}")
    sys.exit(bool(different))


if __name__ == "__main__":
    main()
