"""Serve a local browser webcam preview using the Home Tower pipeline."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from time import monotonic
from uuid import UUID

import cv2
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

if __package__:
    from . import compare_video as video
else:
    import compare_video as video


MAX_BYTES = 512 * 1024
MAX_EDGE = 640
PAGE = Path(__file__).with_name("camera.html").read_bytes()


class CameraPreview:
    def __init__(self, detector):
        self.detection, self.pose = video.load_models((detector, video.MODELS[-1]))
        self.session = None
        self.last_frame = 0

    def annotate(self, session, frame):
        now = monotonic()
        if session != self.session or now - self.last_frame > 3:
            self.session = session
            self.tracker = video.new_tracker()
            self.posture = video.PostureHold(1)
        _, detector, object_classes = self.detection
        _, pose, pose_classes = self.pose
        detection = video.predict_frame(detector, frame, object_classes, self.tracker.args.track_low_thresh)
        people, track_ids = video.track_people(frame, detection, self.tracker)
        person_id = next(index for index, name in detection.names.items() if name == "person")
        furniture = detection[(detection.boxes.cls != person_id) & (detection.boxes.conf >= video.DETECTION_CONFIDENCE)]
        if len(people) == 0:
            self.posture.update({}, now)
            annotated = furniture.plot()
        else:
            joints = video.predict_frame(pose, frame, pose_classes, video.DETECTION_CONFIDENCE)
            annotated, *_ = video.gated_pose_view(furniture.plot(), people, joints, track_ids, self.posture, now)
        self.last_frame = monotonic()
        return annotated


class CameraHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)
        super().setup()

    def allowed_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"localhost:{port}", f"127.0.0.1:{port}"}

    def send_bytes(self, content, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(self), microphone=()")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self.allowed_host():
            self.send_error(403)
        elif self.path == "/":
            self.send_bytes(PAGE, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.allowed_host() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
            self.send_error(403)
            return
        if self.path != "/frame":
            self.send_error(404)
            return
        if self.headers.get_content_type() != "image/jpeg":
            self.send_error(415)
            return
        try:
            session = UUID(self.headers.get("X-Camera-Session", ""))
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(400, "Invalid session or frame length")
            return
        if not 0 < length <= MAX_BYTES:
            self.send_error(413)
            return

        try:
            payload = self.rfile.read(length)
            with JpegImageFile(BytesIO(payload)) as image:
                if max(image.size) > MAX_EDGE:
                    self.send_error(413, "Frame dimensions exceed 640 pixels")
                    return
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, SyntaxError, ValueError):
            self.send_error(400, "Invalid JPEG frame")
            return
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        annotated = self.server.preview.annotate(session, frame)
        ok, jpeg = cv2.imencode(".jpg", annotated)
        if not ok:
            self.send_error(500, "Could not encode preview")
            return
        self.send_bytes(jpeg.tobytes(), "image/jpeg")

    def log_request(self, code="-", size="-"):
        if self.path != "/frame" or str(code) != "200":
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--detector", choices=video.MODELS[:2], default=video.POSE_GATE_MODEL)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    with HTTPServer(("127.0.0.1", args.port), CameraHandler) as server:
        server.preview = CameraPreview(args.detector)
        print(f"Home Tower is ready at http://localhost:{args.port}/ ({args.detector})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
