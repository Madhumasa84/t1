"""Score a YOLO11 detector on the same saved COCO validation sample."""

from argparse import ArgumentParser
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from ultralytics import YOLO
from ultralytics.data.converter import coco80_to_coco91_class


FOLDER = Path(__file__).resolve().parent
SAMPLE = FOLDER.parent / "datasets" / "t1_coco50"
OUTPUT = FOLDER / "coco50_predictions"
FULL_SAMPLE = FOLDER.parent / "datasets" / "t1_coco2017" / "val2017"
COMPARISON = FOLDER / "coco2017_comparison"
MODELS = ("yolo11n", "yolo11s", "yolo26n", "yolo26s")
CONFIDENCE = 0.25
OVERLAP = 0.50


def overlap(prediction, annotation, crowd=False):
    x1, y1, x2, y2 = prediction
    x, y, width, height = annotation
    intersection = max(0, min(x2, x + width) - max(x1, x)) * max(
        0, min(y2, y + height) - max(y1, y)
    )
    area = (x2 - x1) * (y2 - y1)
    denominator = area if crowd else area + width * height - intersection
    return intersection / denominator if denominator > 0 else 0


def score_image(predictions, annotations, counts):
    remaining = defaultdict(list)
    crowds = defaultdict(list)
    for annotation in annotations:
        category = annotation["category_id"]
        target = crowds if annotation["iscrowd"] else remaining
        target[category].append(annotation["bbox"])

    for score, category, box in sorted(predictions, key=lambda row: row[0], reverse=True):
        candidates = remaining[category]
        best = max(range(len(candidates)), key=lambda index: overlap(box, candidates[index]), default=None)
        if best is not None and overlap(box, candidates[best]) >= OVERLAP:
            counts[category]["correct"] += 1
            candidates.pop(best)
        elif any(overlap(box, crowd, crowd=True) >= OVERLAP for crowd in crowds[category]):
            counts[category]["ignored"] += 1
        else:
            counts[category]["extra"] += 1

    for category, boxes in remaining.items():
        counts[category]["missed"] += len(boxes)


def summary(name, counts):
    correct, extra, missed = (counts[key] for key in ("correct", "extra", "missed"))
    return {
        "group": name,
        "correct": correct,
        "extra": extra,
        "missed": missed,
        "crowd_ignored": counts["ignored"],
        "precision": correct / (correct + extra) if correct + extra else None,
        "recall": correct / (correct + missed) if correct + missed else None,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="yolo11n")
    parser.add_argument("--full", action="store_true", help="Evaluate all 5,000 validation images")
    args = parser.parse_args()
    model_name = args.model
    output = OUTPUT if model_name == "yolo11n" else FOLDER / f"coco50_predictions_{model_name}"
    sample_folder = SAMPLE
    if args.full:
        sample_folder = FULL_SAMPLE
        output = COMPARISON / model_name
    evaluate(model_name, sample_folder, output, expected_count=5000 if args.full else 50)


def evaluate(model_name, sample_folder, output, expected_count):
    checkpoint = FOLDER.parent / "weights" / f"{model_name}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sample = json.loads((sample_folder / "sample.json").read_text())
    images = sample["images"]
    if len(images) != expected_count or len({image["id"] for image in images}) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct COCO images")
    annotations = defaultdict(list)
    for row in sample["annotations"]:
        annotations[row["image_id"]].append(row)
    category_ids = coco80_to_coco91_class()
    categories = {row["id"]: row["name"] for row in sample["categories"]}
    person_id = next(key for key, name in categories.items() if name == "person")
    if set(category_ids) != set(categories):
        raise ValueError("The model class mapping does not match the COCO categories")

    paths = [sample_folder / image["file_name"] for image in images]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(checkpoint)
    totals = defaultdict(Counter)
    records = []
    inference_times = []
    for image, path in zip(images, paths, strict=True):
        result = model.predict(
            path, imgsz=480, conf=CONFIDENCE, device="cpu", verbose=False,
        )[0]
        inference_times.append(result.speed["inference"])
        if Path(result.path).name != path.name or result.orig_shape != (image["height"], image["width"]):
            raise ValueError(f"Prediction does not match COCO image {path.name}")
        predictions = [
            (score, category_ids[int(class_id)], [x1, y1, x2, y2])
            for x1, y1, x2, y2, score, class_id in result.boxes.data.tolist()
        ]
        per_class = defaultdict(Counter)
        score_image(predictions, annotations[image["id"]], per_class)
        for category, counts in per_class.items():
            totals[category].update(counts)
        records.append({
            "image": path.name,
            "people": summary("person", per_class[person_id]),
            "other_objects": summary("other objects", sum(
                (counts for category, counts in per_class.items() if category != person_id), Counter()
            )),
        })
        result.save(filename=str(output / path.name))
        if len(records) % 250 == 0:
            print(f"{model_name}: {len(records)}/{expected_count} images", flush=True)

    other = sum((counts for category, counts in totals.items() if category != person_id), Counter())
    furniture = sum(
        (totals[key] for key, name in categories.items() if name in {"chair", "couch", "bed", "dining table"}),
        Counter(),
    )
    rows = [
        summary("people", totals[person_id]),
        summary("all non-person objects", other),
        summary("four furniture classes (included in non-person objects)", furniture),
    ]
    per_class_rows = [
        summary(name, totals[key]) for key, name in categories.items() if totals[key]
    ]
    with (output / "per_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    report = {
        "dataset": "COCO 2017 validation",
        "seed": sample.get("seed"),
        "image_count": len(records),
        "model": str(checkpoint.relative_to(FOLDER.parent)),
        "device": "cpu",
        "image_size": 480,
        "median_inference_ms_excluding_first_image": median(inference_times[1:]),
        "confidence_cutoff": CONFIDENCE,
        "matching_iou": OVERLAP,
        "summary": rows,
        "per_image": records,
    }
    (output / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
