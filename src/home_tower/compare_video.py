"""Preview a home camera or process saved video with the same pipeline."""

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import copy
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT, YAML
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import Annotator, colors

if __package__:
    from .presence import DEFAULT_GRACE, LIVE_GAP_SECONDS, UNAVAILABLE, RoomPresence, grace_seconds
    from .posture import (HOLD_SECONDS, JOINT_CONFIDENCE, SITTING, UNCLEAR, PostureHold, PostureIntervals,
                           estimate_posture)
else:
    from presence import DEFAULT_GRACE, LIVE_GAP_SECONDS, UNAVAILABLE, RoomPresence, grace_seconds
    from posture import (HOLD_SECONDS, JOINT_CONFIDENCE, SITTING, UNCLEAR, PostureHold, PostureIntervals,
                         estimate_posture)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("yolo26n", "yolo26s", "yolo11n-pose")
CLASSES = {"person", "chair", "couch", "bed", "dining table"}
POSE_GATE_MODEL = "yolo26s"
POSE_MATCH_IOU = .50
MIN_JOINTS = 4
DETECTION_CONFIDENCE = .25
PREVIEW_WINDOW = "Home Tower preview"


def load_models(names):
    models = []
    for name in names:
        weights = PROJECT_ROOT.parent / "weights" / f"{name}.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        model = YOLO(weights)
        classes = {"person"} if model.task == "pose" else CLASSES
        class_ids = [index for index, label in model.names.items() if label in classes]
        models.append((name, model, class_ids))
    return models


def new_tracker():
    config = YAML.load(ULTRALYTICS_ROOT / "cfg/trackers/bytetrack.yaml")
    # Confidence already gates detections. Match their motion even when a pose change lowers the score.
    config["fuse_score"] = False
    return BYTETracker(SimpleNamespace(**config))


def predict_frame(model, frame, class_ids, confidence):
    return model.predict(frame, imgsz=480, conf=confidence, classes=class_ids,
                         device="cpu", verbose=False)[0]


def plot_detections(result):
    """Show a generic table label for COCO's only table category."""
    view = copy(result)
    view.names = {index: "table" if name == "dining table" else name
                  for index, name in result.names.items()}
    return view.plot()


def track_people(frame, result, tracker):
    person_id = next(index for index, label in result.names.items() if label == "person")
    candidates = result[result.boxes.cls == person_id]
    tracks = tracker.update(candidates.boxes.cpu().numpy(), frame)
    by_detection = {int(track[-1]): int(track[4]) for track in tracks}
    kept = [index for index, score in enumerate(candidates.boxes.conf.tolist())
            if score >= DETECTION_CONFIDENCE or index in by_detection]
    track_ids = {index: by_detection[detection] for index, detection in enumerate(kept)
                 if detection in by_detection}
    return candidates[kept], track_ids


def preview_ids(track_ids, previous):
    """Keep visible tracks' numbers and give new tracks the lowest free number."""
    current = {track: previous[track] for track in track_ids.values() if track in previous}
    used = set(current.values())
    available = (number for number in range(1, len(track_ids) + 1) if number not in used)
    for track in track_ids.values():
        if track not in current:
            current[track] = next(available)
    return current


def clock(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


def gated_pose_view(frame, people, pose, track_ids, posture_hold, frame_index, display_ids=None,
                    intervals=None, timestamp=None):
    """Match each usable pose to a distinct detector person box.

    With an interval recorder, record the final labels at timestamp and show each track's sitting total.
    """
    confidence = pose.keypoints.conf
    usable = (confidence >= JOINT_CONFIDENCE).sum(dim=1) >= MIN_JOINTS
    poses = pose[usable]
    matches = {}
    if len(people) and len(poses):
        iou = box_iou(people.boxes.xyxy, poses.boxes.xyxy).cpu().numpy()
        valid = iou >= POSE_MATCH_IOU
        # Prioritize the number of valid matches, then how closely the boxes overlap.
        scores = valid * (min(iou.shape) + 1 + iou)
        for person, joint_pose in zip(*linear_sum_assignment(scores, maximize=True)):
            if valid[person, joint_pose]:
                matches[person] = joint_pose

    if matches:
        # Ultralytics draws joints from 0.25 by default; hide weaker joints here.
        selected = poses[list(matches.values())]
        selected.keypoints.data[selected.keypoints.conf < JOINT_CONFIDENCE] = 0
        view = selected.plot(img=frame, boxes=False)
    else:
        view = frame.copy()
    raw_postures = dict.fromkeys(track_ids.values(), UNCLEAR)
    for person, joint_pose in matches.items():
        if person in track_ids:
            raw_postures[track_ids[person]] = estimate_posture(
                poses.keypoints.data[joint_pose].tolist(), poses.boxes.xyxy[joint_pose].tolist()
            )
    postures = posture_hold.update(raw_postures, frame_index)
    if intervals is not None:
        intervals.observe(postures, timestamp)
    annotator = Annotator(view, line_width=2)
    unclear_people = 0
    for index, box in enumerate(people.boxes.xyxy.int().tolist()):
        track_id = track_ids.get(index)
        number = track_id
        label = "Untracked"
        if track_id is not None:
            if display_ids is None:
                label = f"Track {track_id}"
            else:
                number = display_ids[track_id]
                label = f"ID {number}"
        posture = postures.get(track_id, UNCLEAR)
        unclear_people += posture == UNCLEAR
        label += f" - {posture}" if posture != UNCLEAR else " - posture unclear"
        if intervals is not None and track_id is not None:
            label += f" | sat {clock(intervals.seconds(track_id, SITTING, timestamp))}"
        color = colors(number, bgr=True) if number is not None else (0, 215, 255)
        annotator.box_label(box, label, color=color)
    return view, len(poses), len(matches), unclear_people, postures


def output_paths(source, output, pipeline):
    suffix = "annotated" if pipeline else "comparison"
    video = output / f"{source.stem}_{suffix}.mp4"
    return video, video.with_suffix(".partial.mp4"), output / f"{source.stem}_scores.json"


def check_output_paths(sources, output, pipeline):
    # Resolve symlinks and check existing hard links before opening any output.
    input_paths = {path.resolve() for path in sources}
    existing_inputs = [path for path in input_paths if path.exists()]
    for source in sources:
        for path in output_paths(source, output, pipeline):
            if path.resolve() in input_paths or (
                path.exists() and any(path.samefile(input_path) for input_path in existing_inputs)
            ):
                raise ValueError(f"Output would overwrite an input: {path}")


def compare(source, output, models, *, pipeline=False, preview=False, detector=POSE_GATE_MODEL,
            vacancy_grace=DEFAULT_GRACE):
    camera = isinstance(source, int)
    video = temporary = report_path = None
    if not preview:
        check_output_paths([source], output, pipeline)
        video, temporary, report_path = output_paths(source, output, pipeline)
    capture = cv2.VideoCapture(source if camera else str(source))
    window_open = False
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {'camera ' if camera else ''}{source}")
        if camera:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read a frame from {source}")
        if preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            window_open = True
        if not camera and (not math.isfinite(fps) or fps <= 0):
            raise ValueError(f"Cannot read the frame rate from {source}")
        height, width = frame.shape[:2]
        width, height = 640, max(2, 2 * round(height * 640 / width / 2))
        panel_height = height + 28
        times = {name: [] for name, _, _ in models}
        person_frames = dict.fromkeys(times, 0)
        furniture = {name: set() for name in times if name != MODELS[-1]}
        pose_counts = dict.fromkeys(("raw_usable_pose_frames", "shown_pose_frames",
                                    "posture_unclear_frames", "suppressed_pose_boxes",
                                    "suppressed_usable_pose_boxes", "pose_only_frames"), 0)
        frames = 0
        presence = RoomPresence(vacancy_grace)
        presence_events = []
        last_timestamp = -1.0
        tracker = new_tracker()
        display_ids = {} if preview else None
        # The live preview uses elapsed seconds because processing can run below camera FPS.
        posture_hold = PostureHold(1 if preview else fps)
        intervals = PostureIntervals()
        started = monotonic()
        posture_counts = {}
        track_segments = {}
        active_id_counts = Counter()
        detected_without_track = 0
        render_width, render_height = (width, height) if pipeline else (width * 2, panel_height * 2)
        panel = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        command = [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{render_width}x{render_height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-threads", "2", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ] if temporary is not None else None
        try:
            sink = subprocess.Popen(command, stdin=subprocess.PIPE) if command else nullcontext()
            with sink as encoder:
                while ok:
                    observed_at = monotonic()
                    if camera:
                        timestamp = observed_at - started
                    else:
                        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= last_timestamp:
                            timestamp = last_timestamp + 1 / fps if frames else 0.0
                    gap_limit = LIVE_GAP_SECONDS if camera else max(LIVE_GAP_SECONDS, 2 / fps)
                    if frames and timestamp - last_timestamp > gap_limit:
                        presence.disconnect()
                        intervals.close()
                        if not preview:
                            presence_events.append({"seconds": last_timestamp + gap_limit, "state": UNAVAILABLE})
                    last_timestamp = timestamp
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    views = [("Original", frame)]
                    for name, model, class_ids in models:
                        confidence = tracker.args.track_low_thresh if name == detector else DETECTION_CONFIDENCE
                        result = predict_frame(model, frame, class_ids, confidence)
                        times[name].append(result.speed["inference"])
                        strong = result[result.boxes.conf >= DETECTION_CONFIDENCE]
                        labels = {result.names[int(index)] for index in strong.boxes.cls.tolist()}
                        person_frames[name] += "person" in labels
                        if name in furniture:
                            furniture[name].update(labels - {"person"})
                        if name == detector:
                            gate_people, track_ids = track_people(frame, result, tracker)
                            previous_state = presence.state
                            presence.observe(len(gate_people), timestamp)
                            if not preview and presence.state != previous_state:
                                presence_events.append({"seconds": timestamp, "state": presence.state})
                            if preview:
                                display_ids = preview_ids(track_ids, display_ids)
                            if pipeline:
                                person_id = next(index for index, label in result.names.items() if label == "person")
                                background = plot_detections(strong[strong.boxes.cls != person_id])
                            active_id_counts[len(track_ids)] += 1
                            detected_without_track += bool(len(gate_people) and not track_ids)
                            for track_id in track_ids.values():
                                segments = track_segments.setdefault(track_id, [])
                                if segments and segments[-1][1] == frames - 1:
                                    segments[-1][1] = frames
                                else:
                                    segments.append([frames, frames])
                        if name == MODELS[-1]:
                            view, usable, shown, unclear, postures = gated_pose_view(
                                background if pipeline else frame, gate_people, result, track_ids, posture_hold,
                                observed_at if preview else frames, display_ids, intervals, timestamp,
                            )
                            for track_id, posture in postures.items():
                                posture_counts.setdefault(track_id, Counter())[posture] += 1
                            pose_counts["raw_usable_pose_frames"] += bool(usable)
                            pose_counts["shown_pose_frames"] += bool(shown)
                            pose_counts["posture_unclear_frames"] += bool(unclear)
                            pose_counts["suppressed_pose_boxes"] += len(result.boxes) - shown
                            pose_counts["suppressed_usable_pose_boxes"] += usable - shown
                            pose_counts["pose_only_frames"] += bool(len(result.boxes) and not len(gate_people))
                            cv2.putText(view, presence.state, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                        .65, (0, 0, 0), 4, cv2.LINE_AA)
                            cv2.putText(view, presence.state, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                        .65, (255, 255, 255), 1, cv2.LINE_AA)
                            views.append(("Tracked posture", view))
                        elif not pipeline:
                            views.append((name, plot_detections(strong)))
                    if not pipeline:
                        for index, (label, image) in enumerate(views):
                            row, column = divmod(index, 2)
                            x, y = column * width, row * panel_height
                            panel[y+28:y+panel_height, x:x+width] = image
                            cv2.putText(panel, label, (x+8, y+20), cv2.FONT_HERSHEY_SIMPLEX,
                                        .55, (255, 255, 255), 1, cv2.LINE_AA)
                    frames += 1
                    if preview:
                        rate = frames / max(monotonic() - started, .001)
                        cv2.putText(view, f"{detector.upper()} | {rate:.1f} FPS | Q to quit",
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4)
                        cv2.putText(view, f"{detector.upper()} | {rate:.1f} FPS | Q to quit",
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                        cv2.imshow(PREVIEW_WINDOW, view)
                        if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                            break
                        if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    else:
                        encoder.stdin.write((view if pipeline else panel).tobytes())
                    if frames % 100 == 0:
                        source_name = f"camera {source}" if camera else source.name
                        print(f"{source_name}: processed {frames} frames", flush=True)
                    ok, frame = capture.read()
                    if camera and not ok:
                        presence.disconnect()
                        print(presence.state, flush=True)
                # A recording's last frame lasts one frame period. Live coverage ends at the last observation.
                intervals.close(None if camera else last_timestamp + 1 / fps)
                if encoder is not None:
                    encoder.stdin.close()
                    if encoder.wait():
                        raise RuntimeError(f"FFmpeg failed to encode {source.name}")
            if temporary is not None:
                temporary.replace(video)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        capture.release()
        if window_open:
            cv2.destroyAllWindows()
    if preview:
        print(f"Previewed {frames} frames at {frames / max(monotonic() - started, .001):.1f} FPS")
        return
    report = {
        "source": source.name,
        "output": video.name,
        "frames": frames,
        "fps": fps,
        "room_presence": {"vacancy_grace_seconds": vacancy_grace, "events": presence_events},
        "person_prediction_frames": person_frames,
        "tracking": {
            "detector": detector,
            "tracker": "ByteTrack",
            "frames_by_active_id_count": active_id_counts,
            "inclusive_zero_based_frame_ranges_by_id": track_segments,
            "person_detection_frames_without_any_active_id": detected_without_track,
        },
        "posture": {"hold_seconds": HOLD_SECONDS, "frame_counts_by_track_id": posture_counts},
        "sitting_time_by_track_id": intervals.report(),
        "pose_gate": {
            "detector": detector, "minimum_box_iou": POSE_MATCH_IOU,
            "minimum_joint_confidence": JOINT_CONFIDENCE, "minimum_joints": MIN_JOINTS,
            **pose_counts,
        },
        "furniture_classes_predicted": {key: sorted(value) for key, value in furniture.items()},
        "median_cpu_inference_ms": {
            name: median(values[1:] or values) for name, values in times.items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path)
    preview_mode = parser.add_mutually_exclusive_group()
    preview_mode.add_argument("--camera", type=int, metavar="INDEX", help="Preview a camera without recording")
    preview_mode.add_argument("--preview", action="store_true", help="Preview one saved video without recording")
    parser.add_argument("--pipeline", action="store_true", help="Write one annotated view using YOLO26s and pose")
    parser.add_argument("--detector", choices=MODELS[:2], default=POSE_GATE_MODEL,
                        help="Person and furniture detector for single-view processing")
    parser.add_argument("--vacancy-grace", type=grace_seconds, default=DEFAULT_GRACE, metavar="SECONDS")
    args = parser.parse_args()
    preview = args.preview or args.camera is not None
    if args.camera is not None and (args.camera < 0 or args.videos):
        parser.error("--camera requires a nonnegative index and no video paths")
    if args.preview and len(args.videos) != 1:
        parser.error("--preview requires exactly one video path")
    if not preview and not args.videos:
        parser.error("Provide video paths or --camera INDEX")
    if preview and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("Preview needs a graphical display. Run from a desktop terminal.")
    if preview and hasattr(cv2, "currentUIFramework") and not cv2.currentUIFramework():
        parser.error("Preview needs an OpenCV build with window support.")
    if len({video.stem for video in args.videos}) != len(args.videos):
        parser.error("Video filenames must have distinct stems")
    pipeline = args.pipeline or preview
    if not pipeline and args.detector != POSE_GATE_MODEL:
        parser.error("--detector is available for single-view processing and previews")
    output = None
    if not preview:
        output = PROJECT_ROOT / ("pipeline_results" if pipeline else "video_posture")
        if args.detector != POSE_GATE_MODEL:
            output = output / args.detector
        try:
            check_output_paths(args.videos, output, pipeline)
        except ValueError as error:
            parser.error(str(error))
        output.mkdir(parents=True, exist_ok=True)
    models = load_models((args.detector, MODELS[-1]) if pipeline else MODELS)
    sources = [args.camera] if args.camera is not None else args.videos
    for source in sources:
        try:
            compare(source, output, models, pipeline=pipeline, preview=preview, detector=args.detector,
                    vacancy_grace=args.vacancy_grace)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
