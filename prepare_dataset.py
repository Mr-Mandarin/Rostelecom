#!/usr/bin/env python3
"""Подготовка YOLO-датасета из VOC XML аннотаций. """
import json
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path


# === ПУТИ (адаптировано под вашу структуру) ===
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_IMAGES = SCRIPT_DIR / "eval" / "AllFotoAndAnnotations"
SRC_ANNOTS = SCRIPT_DIR / "eval" / "AllFotoAndAnnotations"
YOLO_OUT = SCRIPT_DIR / "data" / "processed" / "yolo_speed"

# === КЛАССЫ (только 3!) ===

CLASS_MAP = {"Download": 0, "Upload": 1, "qms": 2}
CLASS_NAMES = ["Download", "Upload", "qms"]

# === НАСТРОЙКИ ===
SEED = 42
RATIOS = (0.7, 0.2, 0.1)  # train / val / test
REBUILD_DATASET = True  # True = пересоздать, False = использовать кэш


def _parse_voc_to_yolo(xml_path, images_dir):
    """Конвертация VOC XML → YOLO label string."""
    try:
        root = ET.parse(xml_path).getroot()
        filename = root.find('filename').text
        
        if not (Path(images_dir) / filename).exists():
            return None
        
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
        
        return (filename, '\n'.join(lines)) if lines else None
    except Exception as e:
        print(f"  ⚠ Ошибка {xml_path.name}: {e}")
        return None


def _split(items, seed, ratios):
    """
    Перемешать и разделить items на train/val/test.
    
    Args:
        items: список элементов для разделения
        seed: seed для воспроизводимости
        ratios: кортеж (train_ratio, val_ratio, test_ratio)
    
    Returns:
        dict с ключами 'train', 'val', 'test'
    """
    rng = random.Random(seed)
    items = items[:]  # копия списка
    rng.shuffle(items)
    
    n = len(items)
    i1 = int(n * ratios[0])              # граница train
    i2 = i1 + int(n * ratios[1])         # граница val
    
    return {
        'train': items[:i1],
        'val': items[i1:i2],
        'test': items[i2:]
    }


def prepare_yolo_dataset():
    """Основная функция подготовки датасета."""
    yaml_path = YOLO_OUT / "dataset.yaml"
    splits_path = YOLO_OUT / "splits.json"
    
    print("=" * 60)
    print("Подготовка YOLO-датасета")
    print("=" * 60)
    print(f"Источник изображений: {SRC_IMAGES}")
    print(f"Источник аннотаций: {SRC_ANNOTS}")
    print(f"Выходная папка: {YOLO_OUT}")
    print()
    
    # Проверка кэша
    if not REBUILD_DATASET and yaml_path.exists() and splits_path.exists():
        print(f"✓ Используем кэшированный датасет: {YOLO_OUT}")
        return
    
    # Создаём структуру папок
    for split in ["train", "val", "test"]:
        (YOLO_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Парсинг XML
    print("📁 Поиск XML-файлов...")
    xmls = list(SRC_ANNOTS.glob("*.xml"))
    print(f"  Найдено: {len(xmls)} файлов")
    
    print("\n🔄 Конвертация VOC → YOLO...")
    pairs = []
    for xml in xmls:
        result = _parse_voc_to_yolo(xml, SRC_IMAGES)
        if result:
            pairs.append(result)
    
    print(f"  Успешно: {len(pairs)} из {len(xmls)}")
    
    if not pairs:
        print("❌ Нет данных для обработки!")
        return
    
    # Разделение на train/val/test
    print(f"\n✂️  Разделение (seed={SEED}, ratios={RATIOS})...")
    splits = _split(pairs, SEED, RATIOS)
    
    # Копирование файлов
    print("\n📋 Копирование файлов...")
    for split_name, items in splits.items():
        for fname, lbl in items:
            # Копируем изображение
            src_img = SRC_IMAGES / fname
            dst_img = YOLO_OUT / "images" / split_name / fname
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Сохраняем аннотацию
            lbl_path = YOLO_OUT / "labels" / split_name / (Path(fname).stem + ".txt")
            lbl_path.write_text(lbl)
    
    # Создаём dataset.yaml
    print("\n📝 Создание dataset.yaml...")
    
    # Используем относительный путь для локальной работы
    rel_path = YOLO_OUT.relative_to(SCRIPT_DIR) if YOLO_OUT.is_relative_to(SCRIPT_DIR) else YOLO_OUT.absolute()
    
    yaml_content = f"""path: {rel_path}
train: images/train
val: images/val
test: images/test
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(yaml_content)
    
    # Сохраняем splits.json
    splits_data = {
        'seed': SEED,
        'ratios': RATIOS,
        'splits': {k: [p[0] for p in v] for k, v in splits.items()}
    }
    splits_path.write_text(json.dumps(splits_data, indent=2))
    
    # Итоги
    print("\n" + "=" * 60)
    print("✅ Готово!")
    print("=" * 60)
    print(f"  train: {len(splits['train'])} изображений")
    print(f"  val:   {len(splits['val'])} изображений")
    print(f"  test:  {len(splits['test'])} изображений")
    print(f"\n📁 Датасет: {YOLO_OUT}")
    print(f"📄 Config: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_yolo_dataset()