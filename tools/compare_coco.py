"""Prepare the COCO validation archives or combine the finished model runs."""

import argparse
import csv
import json
import shutil
from zipfile import ZipFile

from PIL import Image, ImageDraw

if __package__:
    from .evaluate import COMPARISON, FULL_SAMPLE, MODELS
else:
    from evaluate import COMPARISON, FULL_SAMPLE, MODELS


def prepare():
    archive_folder = FULL_SAMPLE.parent
    with ZipFile(archive_folder / "annotations.zip") as archive:
        data = json.loads(archive.read("annotations/instances_val2017.json"))
    images = sorted(data["images"], key=lambda image: image["id"])
    if len(images) != 5000 or len({image["id"] for image in images}) != 5000:
        raise ValueError("Expected all 5,000 COCO validation images")
    FULL_SAMPLE.mkdir(exist_ok=True)
    with ZipFile(archive_folder / "val2017.zip") as archive:
        for image in images:
            filename = f"{image['id']:012d}.jpg"
            if image["file_name"] != filename:
                raise ValueError(f"Unexpected COCO filename: {image['file_name']}")
            with archive.open(f"val2017/{filename}") as source, (FULL_SAMPLE / filename).open("wb") as target:
                shutil.copyfileobj(source, target)
    manifest = {
        "images": images,
        "categories": data["categories"],
        "annotations": [
            {key: row[key] for key in ("image_id", "category_id", "bbox", "iscrowd")}
            for row in data["annotations"]
        ],
    }
    (FULL_SAMPLE / "sample.json").write_text(json.dumps(manifest) + "\n")
    print(f"Prepared {len(images)} validation images and {len(manifest['annotations'])} annotations")


def combine():
    reports = [json.loads((COMPARISON / model / "scores.json").read_text()) for model in MODELS]
    filenames = [row["image"] for row in reports[0]["per_image"]]
    expected = {row["file_name"] for row in json.loads((FULL_SAMPLE / "sample.json").read_text())["images"]}
    if len(filenames) != 5000 or set(filenames) != expected:
        raise ValueError("The comparison requires the complete validation split")
    for model, report in zip(MODELS, reports, strict=True):
        for key in ("dataset", "device", "image_size", "confidence_cutoff", "matching_iou", "image_count"):
            if reports[0][key] != report[key]:
                raise ValueError(f"{model} used a different setting for {key}")
        if filenames != [row["image"] for row in report["per_image"]]:
            raise ValueError(f"{model} did not evaluate the same images in the same order")
        if {path.name for path in (COMPARISON / model).glob("*.jpg")} != expected:
            raise ValueError(f"{model} is missing prediction images or has extra files")

    paired = COMPARISON / "side_by_side"
    paired.mkdir(exist_ok=True)
    for index, filename in enumerate(filenames, start=1):
        with Image.open(COMPARISON / MODELS[0] / filename) as first:
            width, height = first.size
        row_height = height + 26
        panel = Image.new("RGB", (width * 2, row_height * 2), "#111111")
        draw = ImageDraw.Draw(panel)
        for position, model in enumerate(MODELS):
            row, column = divmod(position, 2)
            x, y = column * width, row * row_height
            with Image.open(COMPARISON / model / filename) as prediction:
                if prediction.size != (width, height):
                    raise ValueError(f"{model} prediction image has a different size for {filename}")
                panel.paste(prediction, (x, y + 26))
            draw.text((x + 8, y + 6), model.upper(), fill="white")
        panel.save(paired / filename, quality=90)
        if index % 500 == 0:
            print(f"Paired images: {index}/{len(filenames)}", flush=True)

    rows = [{
        "model": report["model"],
        **row,
        "median_cpu_inference_ms": report["median_inference_ms_excluding_first_image"],
    } for report in reports for row in report["summary"]]
    with (COMPARISON / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(filenames)} side-by-side images and comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "combine"))
    if parser.parse_args().action == "prepare":
        prepare()
    else:
        combine()
