import json
from itertools import groupby
from math import isclose
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from home_tower import camera_server
from home_tower import compare_video as video
from home_tower.posture import SITTING, UNCLEAR, PostureIntervals
from home_tower.presence import PRESENT, UNAVAILABLE, PresenceHistory

S, U, STAND = SITTING, UNCLEAR, "standing"
GAP = object()  # a camera gap or disconnect: close at the last observation


def record(observations, end=None):
    recorder = PostureIntervals()
    for now, postures in observations:
        if postures is GAP:
            recorder.close()
        else:
            recorder.observe(postures, now)
    recorder.close(end)
    return recorder


def assert_totals_are_interval_sums(recorder):
    report = recorder.report()
    for track, entry in report.items():
        for posture in (S, U):
            parts = [i["end_seconds"] - i["start_seconds"] for i in entry["intervals"] if i["posture"] == posture]
            assert isclose(entry[f"{posture}_seconds"], sum(parts), abs_tol=1e-9)
            assert isclose(entry[f"{posture}_seconds"], recorder.totals[track, posture], abs_tol=5e-4)
    return report


def intervals(report, track):
    return [(i["posture"], i["start_seconds"], i["end_seconds"]) for i in report[track]["intervals"]]


def test_two_simultaneous_tracks_in_different_postures():
    report = assert_totals_are_interval_sums(record([
        (0.0, {1: S, 2: STAND}),
        (0.5, {1: S, 2: STAND}),
        (1.0, {1: S, 2: U}),
        (2.0, {1: STAND, 2: S}),
    ], end=3.0))
    # Track 1 sits from 0 until it stands at 2. Track 2 is unclear 1-2, then sits until the video ends at 3.
    assert intervals(report, 1) == [(S, 0.0, 2.0)]
    assert intervals(report, 2) == [(U, 1.0, 2.0), (S, 2.0, 3.0)]
    assert (report[1]["sitting_seconds"], report[1]["unclear_seconds"]) == (2.0, 0.0)
    assert (report[2]["sitting_seconds"], report[2]["unclear_seconds"]) == (1.0, 1.0)


def test_unclear_splits_sitting_and_stays_visible():
    report = assert_totals_are_interval_sums(record([
        (0.0, {1: S}), (1.0, {1: S}), (1.5, {1: U}), (2.5, {1: U}), (3.0, {1: S}),
    ], end=4.0))
    assert intervals(report, 1) == [(S, 0.0, 1.5), (U, 1.5, 3.0), (S, 3.0, 4.0)]
    assert report[1]["sitting_seconds"] == 2.5  # 1.5 + 1.0
    assert report[1]["unclear_seconds"] == 1.5


def test_irregular_frame_spacing_uses_timestamps_not_frame_counts():
    recorder = record([(0.0, {1: S}), (0.04, {1: S}), (0.3, {1: S}), (0.31, {1: S}), (1.7, {1: S}),
                       (2.25, {1: STAND})], end=3.0)
    report = assert_totals_are_interval_sums(recorder)
    assert intervals(report, 1) == [(S, 0.0, 2.25)]
    assert report[1]["sitting_seconds"] == 2.25


def test_disappearance_ends_the_interval_at_the_first_frame_without_the_track():
    report = assert_totals_are_interval_sums(record([
        (0.0, {1: S, 2: S}), (1.0, {1: S, 2: S}), (2.0, {2: S}), (3.0, {1: S, 2: S}), (4.0, {1: STAND}),
    ], end=5.0))
    # Track 1 is missing at 2 and back at 3: 0-2 and 3-4. Track 2 sits 0-4 and is gone at 4.
    assert intervals(report, 1) == [(S, 0.0, 2.0), (S, 3.0, 4.0)]
    assert intervals(report, 2) == [(S, 0.0, 4.0)]
    assert (report[1]["sitting_seconds"], report[2]["sitting_seconds"]) == (3.0, 4.0)


def test_disconnect_closes_at_the_last_observation_and_skips_the_gap():
    report = assert_totals_are_interval_sums(record([
        (0.0, {1: S}), (1.0, {1: S}), (2.0, {1: S}), (None, GAP), (10.0, {1: S}), (11.0, {1: S}),
    ]))  # live input: coverage ends at the last observation
    assert intervals(report, 1) == [(S, 0.0, 2.0), (S, 10.0, 11.0)]
    assert report[1]["sitting_seconds"] == 3.0


def test_end_of_video_counts_the_last_frame_period():
    fps = 10
    frames = [(index / fps, {1: S}) for index in range(10)]
    report = assert_totals_are_interval_sums(record(frames, end=frames[-1][0] + 1 / fps))
    assert intervals(report, 1) == [(S, 0.0, 1.0)]


def test_running_total_includes_the_open_interval():
    recorder = PostureIntervals()
    recorder.observe({1: S}, 10.0)
    recorder.observe({1: U}, 12.0)
    recorder.observe({1: S}, 13.0)
    assert recorder.seconds(1, S, 14.5) == 3.5
    assert recorder.seconds(1, U, 14.5) == 1.0
    assert recorder.seconds(2, S, 14.5) == 0


# End-to-end: the same scripted observations through the saved-video pipeline and the browser preview.
BOXES = {1: [10, 10, 60, 90], 2: [100, 10, 150, 90], 5: [10, 10, 60, 90], 7: [100, 10, 150, 90]}
# Irregular spacing, sitting -> unclear -> sitting, a disappearance, a 3.7 s coverage gap and an empty last frame.
SCENE = [
    (0.0, {1: S, 2: STAND}),
    (0.1, {1: S, 2: STAND}),
    (0.35, {1: U, 2: S}),
    (0.5, {1: S, 2: S}),
    (1.2, {1: S}),
    (1.3, {1: S, 2: S}),
    (5.0, {1: S, 2: STAND}),
    (5.4, {}),
]
# Worked by hand: track 1 sits 0-0.35, 0.5-1.3 and 5.0-5.4 (1.55 s) and is unclear 0.35-0.5 (0.15 s).
# Track 2 sits 0.35-1.2 (0.85 s). Its return at 1.3 is the last frame before the gap, so it adds nothing.
EXPECTED = {
    1: [(S, 0.0, 0.35), (U, 0.35, 0.5), (S, 0.5, 1.3), (S, 5.0, 5.4)],
    2: [(S, 0.35, 1.2)],
}


def scripted(frame, tracks, joints=False):
    boxes = [BOXES[track] + [.9, 0] for track in tracks]
    keypoints = None
    if joints:
        keypoints = np.zeros((len(tracks), 17, 3), np.float32)
        for row, box in enumerate(boxes):
            for joint in (5, 6, 11, 12):
                keypoints[row, joint] = (box[0] + 10 + joint, box[1] + 5 * joint, .9)
        keypoints = torch.from_numpy(keypoints)
    return Results(frame, path="scene.jpg", names={0: "person"}, keypoints=keypoints,
                   boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6), speed={"inference": 1})


class PassThroughHold:
    """Use the scripted labels as final labels so both paths see identical observations."""

    def __init__(self, fps):
        pass

    def update(self, observations, frame):
        return dict(observations)


@pytest.fixture
def scene(monkeypatch):
    state = {"labels": {}}

    def predict(model, frame, *args):
        return scripted(frame, list(state["labels"]), joints=model == "pose")

    def track(frame, result, tracker):
        return result, dict(enumerate(state["labels"]))

    def estimate(joints, box):
        return next(state["labels"][t] for t in state["labels"] if BOXES[t][0] == round(box[0]))

    monkeypatch.setattr(video, "predict_frame", predict)
    monkeypatch.setattr(video, "track_people", track)
    monkeypatch.setattr(video, "estimate_posture", estimate)
    monkeypatch.setattr(video, "PostureHold", PassThroughHold)
    labels = []
    monkeypatch.setattr(video.Annotator, "box_label", lambda self, box, label, **kwargs: labels.append(label))
    state["box_labels"] = labels
    return state


def replay(scene, tmp_path, frames, fps=10):
    image = np.zeros((120, 160, 3), np.uint8)

    class Recording:
        index = -1

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            if self.index >= len(frames):
                return False, None
            scene["labels"] = frames[self.index][1]
            return True, image

        def get(self, prop):
            return frames[self.index][0] * 1000 if prop == video.cv2.CAP_PROP_POS_MSEC else fps

        def release(self):
            pass

    source = tmp_path / "scene.mp4"
    with patch.object(video.cv2, "VideoCapture", lambda path: Recording()):
        models = [(video.POSE_GATE_MODEL, "detector", [0]), (video.MODELS[-1], "pose", [0])]
        video.compare(source, tmp_path, models, pipeline=True)
    return json.loads((tmp_path / "scene_scores.json").read_text())


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def live(scene, tmp_path, monkeypatch, frames, session="browser"):
    """Send the frames to the preview server's pipeline, then disconnect. Returns every tracker run."""
    clock = Clock()
    monkeypatch.setattr(camera_server, "monotonic", clock)
    runs = []

    class Recorder(PostureIntervals):
        def __init__(self):
            super().__init__()
            runs.append(self)

    monkeypatch.setattr(video, "PostureIntervals", Recorder)
    history = PresenceHistory(tmp_path / "history.sqlite3")
    models = ((video.POSE_GATE_MODEL, "detector", [0]), (video.MODELS[-1], "pose", [0]))
    with patch.object(camera_server.video, "load_models", return_value=models):
        preview = camera_server.CameraPreview(video.POSE_GATE_MODEL, history, 5)
    image = np.zeros((120, 160, 3), np.uint8)
    for now, labels in frames:
        clock.now = now
        if isinstance(labels, str):  # a new browser session
            session = labels
            continue
        scene["labels"] = labels
        preview.annotate(session, image)
    preview.disconnect()
    return preview, history, runs[1:]  # runs[0] is the idle recorder made at startup


def by_track(rows):
    rows = sorted(rows, key=lambda row: (row[0], row[2]))  # by track, then start time
    return {track: [row[1:] for row in group] for track, group in groupby(rows, key=lambda row: row[0])}


def test_replay_and_live_agree_on_the_same_observations(scene, tmp_path, monkeypatch):
    report = replay(scene, tmp_path, SCENE)
    recorded = report["sitting_time_by_track_id"]
    replayed = {int(track): intervals(recorded, track) for track in recorded}
    assert replayed == EXPECTED
    assert (recorded["1"]["sitting_seconds"], recorded["1"]["unclear_seconds"]) == (1.55, 0.15)
    assert (recorded["2"]["sitting_seconds"], recorded["2"]["unclear_seconds"]) == (0.85, 0.0)
    assert report["room_presence"]["events"][1] == {"seconds": 4.3, "state": UNAVAILABLE}  # unchanged gap rule

    _, history, runs = live(scene, tmp_path, monkeypatch, SCENE)
    # The live gap disconnects the camera, so the frames after it start a new tracker run.
    assert len(runs) == 2
    rows = [(track, posture, round(start, 3), round(end, 3)) for run in runs for track, posture, start, end in run.intervals]
    assert by_track(rows) == replayed
    try:
        saved = list(history.db.execute("""SELECT tracker_run, track_id, posture, SUM(seconds) FROM posture_intervals
                                           GROUP BY tracker_run, track_id, posture"""))
        assert len({row[0] for row in saved}) == 2
        assert sorted((track, posture, round(total, 9)) for _, track, posture, total in saved) == sorted(
            (track, posture, round(total, 9)) for run in runs for (track, posture), total in run.totals.items())
        states = [state for state, in history.db.execute("SELECT state FROM presence_events ORDER BY id")]
        assert states == [UNAVAILABLE, PRESENT, UNAVAILABLE, PRESENT, UNAVAILABLE]
    finally:
        history.db.close()


def test_live_totals_survive_the_session_in_sqlite(scene, tmp_path, monkeypatch):
    frames = [(0.0, {1: U}), (0.5, {1: S}), (1.0, {1: S}), (1.6, {1: U}), (2.0, {1: S}), (2.5, {1: S})]
    preview, history, runs = live(scene, tmp_path, monkeypatch, frames)
    # Sitting 0.5-1.6 and 2.0-2.5 (1.6 s). Unclear 0-0.5 and 1.6-2.0 (0.9 s). Disconnect ends coverage at 2.5.
    assert [round(runs[0].totals[1, posture], 9) for posture in (S, U)] == [1.6, 0.9]
    history.db.close()
    reopened = PresenceHistory(tmp_path / "history.sqlite3")
    try:
        totals = dict(reopened.db.execute(
            "SELECT posture, SUM(seconds) FROM posture_intervals WHERE track_id = 1 GROUP BY posture"))
        assert isclose(totals[S], 1.6) and isclose(totals[U], 0.9)
        first = reopened.db.execute("SELECT started_at_utc, ended_at_utc FROM posture_intervals ORDER BY id").fetchone()
        assert first[0].endswith("Z") and first[0] < first[1]
    finally:
        reopened.db.close()


def test_new_session_restarts_ids_without_carrying_time(scene, tmp_path, monkeypatch):
    frames = [(0.0, {1: S}), (1.0, {1: S}), (1.5, "other-browser"), (1.5, {1: S}), (2.0, {1: S})]
    _, history, runs = live(scene, tmp_path, monkeypatch, frames)
    # The first session's track 1 sits 0-1; the second session's track 1 is another tracker run: 1.5-2.0.
    assert [run.intervals for run in runs] == [[(1, S, 0.0, 1.0)], [(1, S, 1.5, 2.0)]]
    try:
        assert len({run for run, in history.db.execute("SELECT tracker_run FROM posture_intervals")}) == 2
    finally:
        history.db.close()


def test_reused_display_id_starts_from_zero(scene, tmp_path, monkeypatch):
    # Track 5 is shown as ID 1 and sits 0-2.5, then leaves. Track 7 arrives at 2.5 and is also shown as ID 1.
    frames = [(0.0, {5: S}), (2.5, {7: S}), (4.0, {7: S})]
    _, history, runs = live(scene, tmp_path, monkeypatch, frames)
    history.db.close()
    assert scene["box_labels"] == ["ID 1 - sitting | sat 0:00", "ID 1 - sitting | sat 0:00", "ID 1 - sitting | sat 0:01"]
    assert runs[0].intervals == [(5, S, 0.0, 2.5), (7, S, 2.5, 4.0)]


def test_clock_format():
    assert [video.clock(seconds) for seconds in (0, 59.9, 65, 3725)] == ["0:00", "0:59", "1:05", "1:02:05"]
