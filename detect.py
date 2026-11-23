from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


folder = Path(__file__).resolve().parent
parser = ArgumentParser(description="Draw furniture detections on a room photo.")
parser.add_argument("photo", nargs="?", default="room.jpg", help="Photo filename inside t1")
photo = folder / parser.parse_args().photo
model = YOLO(folder.parent / "weights" / "yolo11n.pt")

# Keep the four furniture classes this pretrained model recognizes for our project.
furniture = {"chair", "couch", "bed", "dining table"}
class_ids = [class_id for class_id, name in model.names.items() if name in furniture]
result = model.predict(
    photo, classes=class_ids, conf=0.25, imgsz=480, device="cpu", verbose=False
)[0]

for box in result.boxes:
    label = result.names[int(box.cls.item())]
    print(f"{label}: {box.conf.item():.0%} confidence")

if not len(result.boxes):
    print("No furniture detections at this confidence threshold.")

output = folder / ("detected.jpg" if photo == folder / "room.jpg" else f"{photo.stem}_detected.jpg")
result.save(filename=str(output))
print(f"Saved {output}")
