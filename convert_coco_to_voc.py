#!/usr/bin/env python3
"""
Конвертация аннотаций из формата COCO JSON в VOC XML.
Обновляет существующие XML файлы в папках сервисов.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


# Маппинг категорий COCO в VOC
# Оставляем только 3 класса: Download, Upload, qms
COCO_TO_VOC = {
    "download": "Download",
    "upload": "Upload",
    "qms": "qms",
    "web": None  # Игнорируем эту категорию
}

# Список папок сервисов
SERVICE_FOLDERS = ["QMS", "Speedtest", "Table", "Yandex"]


def prettify_xml(elem):
    """Форматирование XML для удобного чтения."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t", encoding='utf-8')


def coco_bbox_to_voc(bbox, img_width, img_height):
    """
    Конвертация bbox из формата COCO [x_min, y_min, width, height]
    в формат VOC [x_min, y_min, x_max, y_max].
    """
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    width = int(bbox[2])
    height = int(bbox[3])
    
    x_max = x_min + width
    y_max = y_min + height
    
    # Ограничиваем координаты размерами изображения
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    
    return x_min, y_min, x_max, y_max


def create_voc_object(cls_name, bbox):
    """Создание XML элемента object."""
    obj = ET.Element("object")
    
    name = ET.SubElement(obj, "name")
    name.text = cls_name
    
    pose = ET.SubElement(obj, "pose")
    pose.text = "Unspecified"
    
    truncated = ET.SubElement(obj, "truncated")
    truncated.text = "0"
    
    difficult = ET.SubElement(obj, "difficult")
    difficult.text = "0"
    
    bndbox = ET.SubElement(obj, "bndbox")
    
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(bbox[0])
    
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(bbox[1])
    
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(bbox[2])
    
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(bbox[3])
    
    return obj


def find_xml_by_filename(services_root, filename):
    """
    Поиск XML файла по имени файла изображения во всех папках сервисов.
    Учитывает особенность именования файлов с суффиксом _png.
    """
    # Получаем базовое имя файла без расширения
    base_name = Path(filename).stem
    
    # Убираем суффикс _png если он есть
    # Пример: 2026-02-03_21-47-13_png -> 2026-02-03_21-47-13
    if base_name.endswith('_png'):
        base_name = base_name[:-4]
    
    for service in SERVICE_FOLDERS:
        service_lower = service.lower()
        annots_dir = services_root / service / f"{service_lower}_annotations"
        if annots_dir.exists():
            for ext in [".xml", ".XML"]:
                xml_path = annots_dir / (base_name + ext)
                if xml_path.exists():
                    return xml_path
    return None


def update_xml_with_coco(xml_path, coco_annotations, img_info, coco_categories):
    """
    Обновление VOC XML файла аннотациями из COCO.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Получаем размеры изображения из XML
    size = root.find('size')
    if size is not None:
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
    else:
        img_width = img_info.get('width', 1000)
        img_height = img_info.get('height', 1000)
    
    # Удаляем все существующие объекты
    for obj in root.findall('object'):
        root.remove(obj)
    
    # Фильтруем аннотации для текущего изображения
    img_id = img_info['id']
    img_annotations = [ann for ann in coco_annotations if ann['image_id'] == img_id]
    
    # Добавляем новые объекты
    for ann in img_annotations:
        category_id = ann['category_id']
        
        # Находим имя категории по id
        cat_name = None
        for cat in coco_categories:
            if cat['id'] == category_id:
                cat_name = cat['name']
                break
        
        if cat_name is None:
            continue
        
        # Маппим категорию
        voc_name = COCO_TO_VOC.get(cat_name)
        if voc_name is None:
            continue  # Пропускаем категорию web
        
        # Конвертируем bbox
        bbox_coco = ann['bbox']
        bbox_voc = coco_bbox_to_voc(bbox_coco, img_width, img_height)
        
        # Создаём и добавляем объект
        obj_elem = create_voc_object(voc_name, bbox_voc)
        root.append(obj_elem)
    
    # Сохраняем XML
    xml_string = prettify_xml(root)
    
    # Убираем декларацию XML от minidom
    lines = xml_string.decode('utf-8').split('\n')
    if lines[0].startswith('<?xml'):
        lines = lines[1:]
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    script_dir = Path(__file__).parent.resolve()
    
    # Пути к файлам
    coco_json_path = script_dir / "eval" / "_annotations.coco.json"
    services_root = script_dir / "eval" / "by_service"
    
    if not coco_json_path.exists():
        print(f"Ошибка: COCO JSON не найден: {coco_json_path}")
        return
    
    if not services_root.exists():
        print(f"Ошибка: Папка сервисов не найдена: {services_root}")
        return
    
    # Загружаем COCO JSON
    print("Загрузка COCO JSON...")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    coco_categories = coco_data['categories']
    coco_images = coco_data['images']
    coco_annotations = coco_data['annotations']
    
    print(f"  Изображений: {len(coco_images)}")
    print(f"  Аннотаций: {len(coco_annotations)}")
    print(f"  Категорий: {len(coco_categories)}")
    print()
    
    # Счётчики
    updated_count = 0
    not_found_count = 0
    skipped_count = 0
    
    # Обрабатываем каждое изображение
    for img_info in coco_images:
        filename = img_info.get('extra', {}).get('name')
        
        if not filename:
            skipped_count += 1
            continue
        
        xml_path = find_xml_by_filename(services_root, filename)
        
        if xml_path is None:
            not_found_count += 1
            print(f"  XML не найден: {filename}")
            continue
        
        try:
            update_xml_with_coco(xml_path, coco_annotations, img_info, coco_categories)
            updated_count += 1
            print(f"  Обновлён: {xml_path.name}")
        except Exception as e:
            print(f"  Ошибка при обновлении {xml_path.name}: {e}")
    
    # Итоги
    print()
    print("=" * 60)
    print("Результаты конвертации:")
    print(f"  Обновлено файлов: {updated_count}")
    print(f"  Не найдено XML: {not_found_count}")
    print(f"  Пропущено: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()