# Rostelecom Speed Recognition

> **Проект**: Автоматическое распознавание скоростей интернета со скриншотов speedtest-сервисов  
> **Цель**: Определить, сделан ли скриншот на сервисе Ростелеком (qms.ru) → `True`, или на другом сервисе → `False`

YOLOv8n + EasyOCR for extracting download/upload speeds from speedtest screenshots.

---

## 📁 Structure

```
speedtest-main/
├── speedtest/
│   ├── main.py                 # Основной скрипт обработки изображений
│   ├── src/
│   │   ├── detector.py         # YOLO детектор областей
│   │   ├── ocr.py              # OCR для распознавания текста
│   │   └── parse.py            # Парсинг числовых значений
│   ├── eval/
│   │   ├── eval.py             # Скрипт оценки качества модели
│   │   └── by_service/         # Тестовые данные по сервисам
│   │       ├── QMS/
│   │       ├── Yandex/
│   │       ├── Speedtest/
│   │       └── Table/
│   ├── data/
│   │   └── processed/yolo_speed/
│   │       └── dataset.yaml    # Конфигурация датасета
│   ├── prepare_dataset.py      # Подготовка датасета (VOC → YOLO)
│   ├── coco_to_voc.py          # Конвертация COCO → VOC
│   └── requirements.txt        # Зависимости
├── weights/                    # Обученные модели (.pt)
└── outputs/                    # Результаты работы скриптов
    ├── results.csv             # CSV с результатами
    └── annotated/              # Изображения с рамками
```

---

## ⚙️ Installation

```bash
# Перейти в папку проекта
cd speedtest

# Установить зависимости
pip install -r requirements.txt

# Или вручную:
pip install ultralytics easyocr opencv-python pandas openpyxl
```

---

## 🚀 Usage

### Inference (Creates output folder by default)

```bash
# Upload local image
python main.py test.jpg

# Upload through URL
python main.py https://url.com/test.jpg

# Folder (recursive)
python main.py path/images/

# List (one path/URL per line)
python main.py path/images.txt

# ZIP archive
python main.py path/images.zip

# Custom output path + log
python main.py test.jpg --outdir out --log out/log.csv
```

**Output:**
- Annotated images: `outputs/annotated/`
- Terminal: `Output dir | Log | Processed: N | Done`
- CSV log: `outputs/results.csv`

**CSV columns:**
```
date | filename | download | unit_download | download_unit_source | 
upload | unit_upload | upload_unit_source | det_conf | ocr_conf | 
qms_detected | qms_conf
```

| Field | Description |
|-------|-------------|
| Speed values | `0` is valid, `null` if not detected |
| Unit sources | `detected` (OCR found), `copied` (from other field), `default` (fallback to Mbit/s) |
| `qms_detected` | `True` if screenshot is from qms.ru, `False` otherwise |

---

## 🗂️ Dataset Preparation

### Convert COCO → VOC (if needed)
```bash
python coco_to_voc.py --input=path/to/coco_annotations --output=path/to/voc_output
```

### Prepare YOLO dataset
```bash
python prepare_dataset.py
```

**What it does:**
1. Scans `eval/AllFotoAndAnnotations` for images and VOC XML annotations
2. Converts VOC (`.xml`) → YOLO format (`.txt`)
3. Splits data: 70% train / 20% val / 10% test
4. Creates `dataset.yaml` configuration

**Example `dataset.yaml`:**
```yaml
path: F:/Downloads/speedtest-main/speedtest/data/processed/yolo_speed
train: images/train
val: images/val
test: images/test
nc: 3
names: ['Download', 'Upload', 'qms']
```

> ⚠️ **Important**: Ensure `nc` (number of classes) and `names` match your annotations!

---

## 🎯 Training

```bash
yolo detect train \
    data=data/processed/yolo_speed/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    name=qms_final \
    project=runs/detect \
    device=cpu
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data` | `dataset.yaml` | Dataset configuration path |
| `model` | `yolov8n.pt` | Pretrained model (n/s/m/l/x) |
| `epochs` | `100` | Training epochs |
| `imgsz` | `640` | Image size for training |
| `batch` | `8` | Batch size (reduce if OOM) |
| `device` | `cpu` | `cpu` or `0` for GPU |

**Outputs:**
- Logs: `runs/detect/qms_final/`
- Plots: `runs/detect/qms_final/results.png`
- Best model: `runs/detect/qms_final/weights/best.pt`

> ⏱️ **Training time**: ~3-5 hours on CPU, ~30-60 min on GPU

---

## 🧪 Testing

```bash
# Test on QMS images (should be qms_detected=True)
python main.py eval/by_service/QMS/qms_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# Test on other services (should be qms_detected=False)
python main.py eval/by_service/Yandex/yandex_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
python main.py eval/by_service/Speedtest/speedtest_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
python main.py eval/by_service/Table/table_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# Test on all images
python main.py eval/AllFotoAndAnnotations --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
```

**Expected results:**
| Service | `qms_detected` |
|---------|---------------|
| QMS (qms.ru) | `True` (~95%) |
| Yandex | `False` |
| Speedtest.net | `False` |
| Table | `False` |

---

## 📈 Evaluation

```bash
cd eval && python eval.py --weights=../runs/detect/runs/detect/qms_final/weights/best.pt
```

**Sample output:**
```
======================================================================
Service               mAP50   mAP50-95  Precision     Recall
======================================================================
QMS                  0.9758     0.7932     0.9446     0.9953
Speedtest            0.8280     0.6383     0.7800     1.0000
Table                0.9950     0.6739     0.9987     1.0000
Yandex               0.9911     0.8111     0.9848     0.9859
======================================================================
```

| Metric | Description | Good value |
|--------|-------------|------------|
| `mAP50` | Mean AP at IoU=0.5 | > 0.7 |
| `mAP50-95` | Mean AP at IoU 0.5-0.95 | > 0.5 |
| `Precision` | Fraction of correct detections | > 0.8 |
| `Recall` | Fraction of objects found | > 0.8 |

---

## 🔄 Pipeline

**Input → YOLO zones → Crop+Pad → OCR variants → Parse → Annotate+Log**

1. Detect Download/Upload boxes (top-1 per class)
2. OCR each crop with preprocessing variants (inverted, threshold)
3. Extract number + unit (Mbit/s, Kbit/s)
4. Fallback: copy unit between fields or default to Mbit/s
5. Save annotated image + append CSV row

**Features:**
- ✅ Russian+English OCR (EasyOCR)
- ✅ Offline inference
- ✅ URL support
- ✅ Auto-detects CUDA/MPS/CPU
- ✅ QMS service detection (`qms_detected` flag)

**Classes:** 
- `0` = Download (🔴 red box)
- `1` = Upload (🔵 blue box)  
- `2` = qms (🟢 green box, service identifier)

---

## ❓ Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: Weights not found` | Wrong model path | Use full path or check `runs/detect/runs/detect/` |
| `qms_detected=False` always | Model not trained on qms class | Check `dataset.yaml`: `nc: 3` and `names: [..., 'qms']` |
| Low OCR accuracy | Poor image quality | Use `preprocess_variants()` in `ocr.py` |
| Slow processing | Running on CPU | Use GPU: `device=0` or reduce `batch` |
| eval.py class mismatch | `CLASS_NAMES` out of sync | Sync `eval.py` with `dataset.yaml` |

---

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone <repo-url>
cd speedtest-main/speedtest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
python prepare_dataset.py

# 4. Train model
yolo detect train data=data/processed/yolo_speed/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 name=qms_final project=runs/detect device=cpu

# 5. Test
python main.py eval/by_service/QMS/qms_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# 6. Evaluate
cd eval && python eval.py --weights=../runs/detect/runs/detect/qms_final/weights/best.pt
```

---

## 📞 Contacts & Support

> **Support for my changes**

| Field | Value |
|-------|-------|
| **Author** | Андрей |
| **Telegram** | @mr_mandarin0 |
| **Date** | Март 2026 |

> 💡 **Tip**: When encountering errors — check file paths and config sync (`dataset.yaml`, `eval.py`, `detector.py`).

---

*Documentation valid for project version from 22.03.2026* 🎯