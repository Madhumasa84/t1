"""Conservative posture estimates from COCO pose joints in a fixed camera view."""

from math import hypot


UNCLEAR = "unclear"
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
