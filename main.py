#!/usr/bin/env python3
"""Speed extraction with YOLO + OCR."""

import os
import warnings

# Suppress warnings before imports
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings("ignore")

import argparse
import csv
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import cv2
import requests

from src.detector import load_detector, detect_regions
from src.ocr import init_ocr, ocr_number_and_unit
from src.parse import parse_speed


def download_url(url: str) -> str:
    """Download URL to temp file."""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    basename = url.split("/")[-1].split("?")[0]
    if not basename.endswith(('.jpg', '.jpeg', '.png')):
        basename = f"{hash(url) % 10**8}.jpg"
    temp_path = Path(tempfile.gettempdir()) / basename
    temp_path.write_bytes(r.content)
    return str(temp_path)


def extract_zip(zip_path: str) -> str:
    """Extract zip to temp dir."""
    temp_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
    return str(temp_dir)


def resolve_inputs(input_arg: str) -> list:
    """
    Resolve input to list of (source, local_path, filename) tuples.
    
    Handles: single file, folder, .txt list, .zip archive
    """
    items = []
    path = Path(input_arg)
    
    # .txt file with paths/URLs
    if path.suffix == '.txt' and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith(('http://', 'https://')):
                try:
                    local = download_url(line)
                    items.append((line, local, Path(local).name))
                except:
                    continue
            else:
                p = Path(line)
                if p.exists():
                    items.append((line, str(p), p.name))
    
    # .zip archive
    elif path.suffix == '.zip' and path.exists():
        temp_dir = extract_zip(str(path))
        for img in Path(temp_dir).rglob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                items.append((str(img), str(img), img.name))
    
    # Folder
    elif path.is_dir():
        for img in path.rglob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                items.append((str(img), str(img), img.name))
    
    # URL
    elif input_arg.startswith(('http://', 'https://')):
        local = download_url(input_arg)
        items.append((input_arg, local, Path(local).name))
    
    # Single file
    elif path.exists():
        items.append((str(path), str(path), path.name))
    
    else:
        raise ValueError(f"Invalid input: {input_arg}")
    
    return items


def extract_speeds(image_path: str, model, reader) -> dict:
    """Extract speeds from single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    
    boxes = detect_regions(model, img)
    
    h, w = img.shape[:2]
    results = {
        "download_value": None,
        "download_unit": "",
        "download_unit_source": "",
        "upload_value": None,
        "upload_unit": "",
        "upload_unit_source": "",
        "det_conf": 0.0,
        "ocr_conf": 0.0,
        "boxes": boxes
    }
    
    det_confs = []
    ocr_confs = []
    
    for cls_name, prefix in [("Download", "download"), ("Upload", "upload")]:
        box_info = boxes.get(cls_name)
        if not box_info:
            continue
        
        # Crop with padding
        x1, y1, x2, y2 = box_info["xyxy"]
        pad = int(max(x2-x1, y2-y1) * 0.05)
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        crop = img[y1:y2, x1:x2]
        
        # OCR
        text, unit, conf = ocr_number_and_unit(reader, crop)
        speed = parse_speed(text)
        
        results[f"{prefix}_value"] = speed
        if unit:
            results[f"{prefix}_unit"] = unit
            results[f"{prefix}_unit_source"] = "detected"
        
        det_confs.append(box_info["conf"])
        if speed is not None:
            ocr_confs.append(conf)
    
    # Unit fallback
    if not results["download_unit"] and results["upload_unit"]:
        results["download_unit"] = results["upload_unit"]
        results["download_unit_source"] = "copied"
    elif not results["upload_unit"] and results["download_unit"]:
        results["upload_unit"] = results["download_unit"]
        results["upload_unit_source"] = "copied"
    
    if not results["download_unit"]:
        results["download_unit"] = "Mbit/s"
        results["download_unit_source"] = "default"
    if not results["upload_unit"]:
        results["upload_unit"] = "Mbit/s"
        results["upload_unit_source"] = "default"
    
    if det_confs:
        results["det_conf"] = min(det_confs)
    if ocr_confs:
        results["ocr_conf"] = min(ocr_confs)

        # === Обработка класса qms ===
    qms_box = boxes.get("qms")
    if qms_box:
        # Модель нашла qms — считаем detected
        results["qms_detected"] = True
        results["qms_conf"] = qms_box["conf"]  # ← Confidence от YOLO, не от OCR
        results["qms_box"] = qms_box
    else:
        results["qms_detected"] = False
        results["qms_conf"] = 0.0
        results["qms_box"] = None
    # === Конец обработки qms ===

    return results


def annotate_image(image_bgr, boxes: dict, values: dict):
    """Draw boxes and values."""
    img = image_bgr.copy()
    colors = {"Download": (0, 0, 255), "Upload": (255, 0, 0), "qms": (0, 255, 0)}  # ← Добавить qms цвет
    
    for cls_name, box_info in boxes.items():
        if not box_info:
            continue
        
        x1, y1, x2, y2 = box_info["xyxy"]
        color = colors.get(cls_name, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # ← Добавить блок для qms
        if cls_name == "qms":
            qms_text = "QMS: detected" if values.get("qms_detected") else "QMS: not found"
            cv2.putText(img, qms_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            continue  # Для qms не рисуем числовое значение
        
        # Существующий код для Download/Upload
        prefix = "download" if cls_name == "Download" else "upload"
        val = values.get(f"{prefix}_value")
        unit = values.get(f"{prefix}_unit", "")
        val_str = "null" if val is None else str(val)
        label = f"{cls_name}: {val_str} {unit}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Batch speed extraction")
    parser.add_argument("input", help="Image file, folder, .txt list, .zip archive, or URL")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log = f"outputs/results_{timestamp}.csv"
    parser.add_argument("--log", default=default_log, help="CSV log path")
    parser.add_argument("--weights", default="weights/best.onnx", help="YOLO weights")
    args = parser.parse_args()
    
    # СТАЛО (красивое имя файла):
    outdir = Path(args.outdir)
    annotated_dir = outdir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # === Красивое имя для CSV ===
    # 1. Форматируем timestamp: 2026-03-20_23-45-30
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. Извлекаем имя папки источника (QMS, Yandex, Speedtest, Table)
    input_path = Path(args.input)
    if input_path.is_dir():
        # Берём имя папки: eval/by_service/QMS/qms_images → QMS
        source_name = input_path.parent.name.upper()
        if source_name in ["QMS_IMAGES", "SPEEDTEST_IMAGES", "TABLE_IMAGES", "YANDEX_IMAGES"]:
            source_name = source_name.replace("_IMAGES", "")
    else:
        source_name = "MIXED"

    # 3. Создаём красивое имя файла: results_QMS_2026-03-20_23-45-30.csv
    log_filename = f"results_{source_name}_{timestamp}.csv"
    log_path = outdir / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📁 Source: {source_name}")
    print(f"📄 Log file: {log_path.name}")
    # === Конец красивого имени ===
    
    # Load models once
    model = load_detector(args.weights)
    reader = init_ocr()
    
    # Resolve inputs
    items = resolve_inputs(args.input)
    
    # Process
    processed = 0
    for source, local_path, filename in items:
        try:
            # Extract
            result = extract_speeds(local_path, model, reader)
            
            # Annotate
            img = cv2.imread(local_path)
            annotated = annotate_image(img, result["boxes"], result)
            out_path = annotated_dir / filename
            cv2.imwrite(str(out_path), annotated)
            
            # Log (update existing row or append)
            new_row = [
                datetime.now().isoformat(),
                filename,
                result["download_value"] if result["download_value"] is not None else "null",
                result["download_unit"],
                result.get("download_unit_source", "default"),
                result["upload_value"] if result["upload_value"] is not None else "null",
                result["upload_unit"],
                result.get("upload_unit_source", "default"),
                f"{result['det_conf']:.4f}",
                f"{result['ocr_conf']:.4f}",
                str(result.get("qms_detected", False)),  
                f"{result.get('qms_conf', 0):.4f}"        
            ]
            
            header = ["date", "filename", "download", "unit_download", "download_unit_source", 
                      "upload", "unit_upload", "upload_unit_source", "det_conf", "ocr_conf",
                      "qms_detected", "qms_conf"]  
            rows = []
            updated = False
            
            if log_path.exists():
                with open(log_path, 'r', newline='') as f:
                    csv_reader = csv.reader(f)
                    rows = list(csv_reader)
                    for i, row in enumerate(rows):
                        if i == 0:
                            continue
                        if len(row) > 1 and row[1] == filename:
                            rows[i] = new_row
                            updated = True
                            break
            
            if not updated:
                if not rows:
                    rows = [header, new_row]
                else:
                    rows.append(new_row)
            
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            processed += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Output summary
    print(f"Output dir: {outdir.resolve()}")
    print(f"Log: {log_path.resolve()}")
    print(f"Processed: {processed}")
    print("Done")


if __name__ == "__main__":
    main()
