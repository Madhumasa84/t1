"""Conservative posture estimates from COCO pose joints in a fixed camera view."""

from collections import Counter
from math import hypot


UNCLEAR = "unclear"
SITTING = "sitting"
RECORDED_POSTURES = (SITTING, UNCLEAR)
JOINT_CONFIDENCE = .30
HOLD_SECONDS = .30


def estimate_posture(joints, box):
    """Describe visible geometry. Walking with an upright body counts as standing."""
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
        sitting |= abs(ky - hy) <= .60 * thigh
        if joints[ankle][2] >= JOINT_CONFIDENCE:
            _, ay = joints[ankle][:2]
            if ky - hy >= .75 * thigh and ay - ky >= .25 * torso:
                # Facing the camera, seated thighs project shorter than both torso and shins.
                if ky - hy <= .70 * torso and ky - hy <= .85 * (ay - ky):
                    shortened_thighs += 1
                else:
                    standing = True
    sitting |= shortened_thighs == 2
    if sitting != standing:
        return "sitting" if sitting else "standing"
    return UNCLEAR


class PostureHold:
    """Require consecutive observations of the same posture for each active ID."""

    def __init__(self, fps):
        self.fps = fps
        self.pending = {}

    def update(self, observations, frame):
        active = {}
        labels = {}
        for track_id, posture in observations.items():
            previous, started = self.pending.get(track_id, (UNCLEAR, frame))
            if posture != previous:
                started = frame
            active[track_id] = (posture, started)
            labels[track_id] = posture if frame - started >= HOLD_SECONDS * self.fps else UNCLEAR
        self.pending = active
        return labels


class PostureIntervals:
    """Turn each track's final posture labels into sitting and unclear intervals.

    Times are seconds on one clock: video timestamps for recordings, elapsed time for live input. An
    observation lasts until the next one, so a track whose label changes, or that is missing from an
    observation, ends its interval at that observation's time. close() ends coverage for every track:
    a camera gap or disconnect closes at the last observation, the end of a video at the end of its
    last frame. Track IDs must be unique for the life of the recorder; use a new recorder per tracker.
    """

    def __init__(self):
        self.intervals = []  # (track, posture, start, end), in the order they ended
        self.open = {}
        self.totals = Counter()
        self.last = None

    def observe(self, postures, now):
        """Record the final label of every visible track at time now."""
        for track, (posture, _) in list(self.open.items()):
            if postures.get(track) != posture:
                self._end(track, now)
        for track, posture in postures.items():
            if posture in RECORDED_POSTURES:
                self.open.setdefault(track, (posture, now))
        self.last = now

    def close(self, end=None):
        """End coverage. Nothing after the last observation counts unless end is given."""
        for track in list(self.open):
            self._end(track, self.last if end is None else end)

    def _end(self, track, end):
        posture, start = self.open.pop(track)
        if end > start:
            self.intervals.append((track, posture, start, end))
            self.totals[track, posture] += end - start

    def seconds(self, track, posture, now):
        """Total time so far, including an interval still open at time now."""
        open_posture, start = self.open.get(track, (None, now))
        return self.totals[track, posture] + (now - start if open_posture == posture else 0)

    def report(self):
        """Closed intervals by track, rounded to milliseconds. Each total is the sum of its intervals."""
        intervals = {}
        for track, posture, start, end in sorted(self.intervals, key=lambda interval: interval[::2]):
            intervals.setdefault(track, []).append((posture, round(start, 3), round(end, 3)))
        report = {}
        for track, rows in intervals.items():
            report[track] = {f"{name}_seconds": round(sum(end - start for posture, start, end in rows if posture == name), 3)
                             for name in RECORDED_POSTURES}
            report[track]["intervals"] = [{"posture": posture, "start_seconds": start, "end_seconds": end}
                                          for posture, start, end in rows]
        return report
