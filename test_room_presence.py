from unittest.mock import patch

import pytest

from camera_server import CameraPreview
from presence import CHECKING, PRESENT, UNAVAILABLE, VACANT, PresenceHistory, RoomPresence


def test_two_people_depart_occlusion_and_reentry():
    room = RoomPresence()
    timeline = [(0, 2), (1, 1), (2, 0), (5, 1), (6, 0), (10.99, 0),
                (11, 0), (12, 0), (13, 1)]
    states = [room.observe(count, timestamp) for timestamp, count in timeline]
    assert states == [PRESENT] * 6 + [VACANT, VACANT, PRESENT]


def test_startup_gap_and_configured_grace():
    room = RoomPresence(2)
    assert room.observe(0, 0) == CHECKING
    room.disconnect()
    assert room.state == UNAVAILABLE
    assert room.observe(0, 100) == CHECKING
    assert room.observe(0, 101.9) == CHECKING
    assert room.observe(0, 102) == VACANT


@pytest.mark.parametrize("grace", [0, -1, float("nan"), float("inf")])
def test_invalid_grace(grace):
    with pytest.raises(ValueError):
        RoomPresence(grace)


def test_idle_disconnect_writes_one_gap_and_history_survives_restart(tmp_path):
    path = tmp_path / "events.sqlite3"
    history = PresenceHistory(path)
    with patch("camera_server.video.load_models", return_value=(None, None)):
        preview = CameraPreview("detector", history, 5)
    preview.session = "laptop"
    preview.last_frame = 3
    history.record(preview.room.observe(1, 2))
    preview.room.observe(0, 3)
    with patch("camera_server.monotonic", return_value=10):
        preview.check_connection()
        preview.check_connection()
    assert preview.room.state == UNAVAILABLE
    history.db.close()
    reopened = PresenceHistory(path)
    try:
        reopened.record(UNAVAILABLE)
        assert list(reopened.db.execute("SELECT state FROM presence_events ORDER BY id")) == [
            (UNAVAILABLE,), (PRESENT,), (UNAVAILABLE,)]
    finally:
        reopened.db.close()
