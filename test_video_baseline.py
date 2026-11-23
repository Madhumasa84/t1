import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

import compare_video as video


@pytest.mark.parametrize("output_name", ["one_annotated.mp4", "one_annotated.partial.mp4", "one_scores.json"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_batch_rejects_input_collision_before_processing(tmp_path, monkeypatch, output_name, alias):
    out = tmp_path / "t1" / "pipeline_results"
    out.mkdir(parents=True)
    first = tmp_path / "one.mp4"
    first.write_bytes(b"first input")
    destination = out / output_name
    second = destination if alias == "direct" else tmp_path / "two.mp4"
    second.write_bytes(b"second input")
    if alias == "symlink":
        destination.symlink_to(second)
    if alias == "hardlink":
        destination.hardlink_to(second)
    before = set(out.iterdir())
    monkeypatch.setattr(video, "ROOT", tmp_path)
    monkeypatch.setattr(video, "load_models", lambda *_: pytest.fail("collision missed"))
    monkeypatch.setattr(sys, "argv", ["compare_video.py", "--pipeline", str(first), str(second)])
    with pytest.raises(SystemExit) as error:
        video.main()
    assert error.value.code == 2
    assert set(out.iterdir()) == before
    assert second.read_bytes() == b"second input"


def result(frame, boxes, joints=None):
    return Results(frame, path="test.jpg", names={0: "person"},
                   boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6),
                   keypoints=None if joints is None else torch.from_numpy(joints),
                   speed={"inference": 1})


def test_unclear_follows_displayed_labels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[0, 0, 40, 90, .9, 0], [50, 0, 90, 90, .9, 0]]
    joints = np.zeros((2, 17, 3), dtype=np.float32)
    for person, x in enumerate((20, 70)):
        for joint, y in [(5, 10), (11, 40), (13, 65), (15, 85)]:
            joints[person, joint] = (x, y, .95)
    people = result(frame, boxes)
    poses = result(frame, boxes, joints)
    hold = video.PostureHold(10)

    def labels(index, tracks):
        _, _, matched, unclear, postures = video.gated_pose_view(frame, people, poses, tracks, hold, index)
        assert matched == 2
        return unclear, postures

    assert labels(0, {0: 1, 1: 2}) == (2, {1: "unclear", 2: "unclear"})
    assert labels(3, {0: 1, 1: 2}) == (0, {1: "standing", 2: "standing"})
    assert labels(4, {0: 1}) == (1, {1: "standing"})


def test_native_preview_survives_progress_log(monkeypatch, capsys):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    class Camera:
        remaining = 101

        def isOpened(self):
            return True

        def read(self):
            self.remaining -= 1
            return self.remaining >= 0, frame

        def set(self, *args):
            pass

        def get(self, prop):
            return 30

        def release(self):
            pass

    monkeypatch.setattr(video.cv2, "VideoCapture", lambda index: Camera())
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(video.cv2, name, lambda *args: None)
    monkeypatch.setattr(video.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(video.cv2, "waitKey", lambda *args: -1)

    def predict(model, frame, *args):
        joints = np.empty((0, 17, 3), np.float32) if model == "pose" else None
        return result(frame, [], joints)

    monkeypatch.setattr(video, "predict_frame", predict)
    models = [(video.POSE_GATE_MODEL, "detector", [0]), (video.MODELS[-1], "pose", [0])]
    video.compare(0, None, models, pipeline=True, preview=True)
    log = capsys.readouterr().out
    assert "camera 0: processed 100 frames" in log
    assert "Previewed 101 frames" in log


@pytest.mark.parametrize("module", [False, True])
def test_camera_entry_points(module):
    project = Path(__file__).resolve().parent
    command = ["-m", f"{project.name}.camera_server"] if module else [str(project / "camera_server.py")]
    run = subprocess.run([sys.executable, *command, "--help"], cwd=project.parent,
                         capture_output=True, text=True, check=True)
    assert "--detector" in run.stdout


def test_front_facing_sitting_and_missing_leg():
    joints = np.zeros((17, 3))
    for idx, xy in {5: (405, 288), 6: (266, 280), 11: (381, 456), 12: (284, 454),
                    13: (425, 563), 14: (273, 560), 15: (425, 706), 16: (243, 696)}.items():
        joints[idx] = (*xy, .95)
    assert video.estimate_posture(joints, [175, 147, 479, 717]) == "sitting"
    joints[16, 2] = 0
    assert video.estimate_posture(joints, [175, 147, 479, 717]) == "unclear"


def test_track_survives_pose_change_confidence_dip_and_short_gap():
    tracker = video.new_tracker()
    frame = np.zeros((200, 200, 3), np.uint8)

    def tracks(boxes):
        return video.track_people(frame, result(frame, boxes), tracker)[1]

    original = tracks([[20, 10, 80, 180, .95, 0]])
    assert len(original) == 1
    seated = [40, 60, 110, 175, .40, 0]
    assert tracks([seated]) == original
    assert tracks([]) == {}
    assert tracks([seated]) == original
    distant = [145, 20, 195, 180, .95, 0]
    tracks([distant])
    different = tracks([distant])
    assert len(different) == 1 and different != original
