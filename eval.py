#!/usr/bin/env python3
"""Evaluate YOLO detector per service."""

import argparse
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO


CLASS_NAMES = ["Download", "Upload", "qms"]
CLASS_MAP = {
    "Download": 0,
    "Upload": 1,
    "qms": 2
}


def voc_to_yolo_label(xml_path: Path) -> str:
    """Convert VOC XML to YOLO label string."""
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find('size')
        w_img = int(size.find('width').text)
        h_img = int(size.find('height').text)
        
        lines = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in CLASS_MAP:
                continue
            
            box = obj.find('bndbox')
            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            xc = ((x1 + x2) / 2) / w_img
            yc = ((y1 + y2) / 2) / h_img
            wb = (x2 - x1) / w_img
            hb = (y2 - y1) / h_img
            
            lines.append(f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {wb:.6f} {hb:.6f}")
        
        return '\n'.join(lines)
    except:
        return ""


def eval_service(model: YOLO, service_dir: Path) -> dict:
    """Evaluate model on a single service dataset."""
    service_name = service_dir.name.lower()
    images_dir = service_dir / f"{service_name}_images"
    annots_dir = service_dir / f"{service_name}_annotations"
    
    if not images_dir.exists() or not annots_dir.exists():
        return None
    
    # Create temp dir with YOLO labels
    temp_dir = Path(tempfile.mkdtemp())
    temp_images = temp_dir / "images"
    temp_labels = temp_dir / "labels"
    temp_images.mkdir()
    temp_labels.mkdir()
    
    # Copy images and convert labels
    for img in images_dir.glob("*.jpg"):
        shutil.copy2(img, temp_images / img.name)
        xml_path = annots_dir / (img.stem + ".xml")
        if xml_path.exists():
            label_txt = voc_to_yolo_label(xml_path)
            if label_txt:
                (temp_labels / (img.stem + ".txt")).write_text(label_txt)
    
    # Create dataset.yaml
    yaml_path = temp_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {temp_dir.absolute()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    
    # Run validation
    try:
        metrics = model.val(data=str(yaml_path), split='val', verbose=False, save=False, plots=False)
        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr
        }
    except:
        return None
    finally:
        # shutil.rmtree(temp_dir, ignore_errors=True)  # временно отключаем
        print(f"Labels saved to: {temp_dir}")  # добавь


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO per service")
    parser.add_argument("--weights", default="../weights/best.pt", help="Model weights")
    parser.add_argument("--services_dir", default="by_service", help="Services root dir")
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.resolve()
    weights_path = (script_dir / args.weights).resolve()
    services_dir = (script_dir / args.services_dir).resolve()
    
    if not services_dir.exists():
        print(f"Services directory not found: {services_dir}")
        return
    
    # Load model
    model = YOLO(str(weights_path))
    
    # Evaluate each service
    
    results = {}
    for service_path in sorted(services_dir.iterdir()):
        if not service_path.is_dir():
            continue
        
        service_name = service_path.name
        print(f"Evaluating {service_name}...", end=" ")
        
        metrics = eval_service(model, service_path)
        if metrics:
            results[service_name] = metrics
            print("✓")
        else:
            print("✗ (skipped)")
    
    # Print table
    if not results:
        print("\nNo valid services found.")
        return
    
    print("\n" + "="*70)
    print(f"{'Service':<20} {'mAP50':>10} {'mAP50-95':>10} {'Precision':>10} {'Recall':>10}")
    print("="*70)
    
    for service, metrics in results.items():
        print(f"{service:<20} "
              f"{metrics['mAP50']:>10.4f} "
              f"{metrics['mAP50-95']:>10.4f} "
              f"{metrics['Precision']:>10.4f} "
              f"{metrics['Recall']:>10.4f}")
    
    print("="*70)


if __name__ == "__main__":
    main()
