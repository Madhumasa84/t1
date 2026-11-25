"""Room presence decisions shared by live cameras and recorded video."""

import math
import sqlite3

PRESENT = "Person present"
VACANT = "No one visible"
UNAVAILABLE = "Camera unavailable"
CHECKING = "Checking room"
DEFAULT_GRACE = 5.0
LIVE_GAP_SECONDS = 3.0


def grace_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("vacancy grace must be finite and positive")
    return seconds


class RoomPresence:
    def __init__(self, grace=DEFAULT_GRACE):
        self.grace = grace_seconds(grace)
        self.state = UNAVAILABLE
        self.empty_since = None

    def observe(self, people, now):
        if people:
            self.empty_since = None
            self.state = PRESENT
        else:
            if self.empty_since is None:
                self.empty_since = now
            if now - self.empty_since >= self.grace:
                self.state = VACANT
            elif self.state == UNAVAILABLE:
                self.state = CHECKING
        return self.state

    def disconnect(self):
        self.empty_since = None
        self.state = UNAVAILABLE


class PresenceHistory:
    """One local camera timeline. Persist state transitions even across restarts."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS presence_events (
            id INTEGER PRIMARY KEY, observed_at_utc TEXT NOT NULL, state TEXT NOT NULL
        )""")

    def record(self, state):
        with self.db:
            self.db.execute("""INSERT INTO presence_events (observed_at_utc, state)
                SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now'), ?
                WHERE (SELECT state FROM presence_events ORDER BY id DESC LIMIT 1) IS NOT ?""",
                            (state, state))
