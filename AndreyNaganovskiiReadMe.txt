# 📖 Документация проекта: Компьютерное зрение в помощь монтажнику

> **Проект**: Автоматическое распознавание скоростей интернета со скриншотов speedtest-сервисов  
> **Цель**: Определить, сделан ли скриншот на сервисе Ростелеком (qms.ru) → `True`, или на другом сервисе → `False`

---

## 📁 Структура проекта

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
│   ├── coco_to_voc.py          # Конвертация COCO → VOC (если есть)
│   └── requirements.txt        # Зависимости
├── weights/                    # Обученные модели (.pt)
└── outputs/                    # Результаты работы скриптов
    ├── results.csv             # CSV с результатами
    └── annotated/              # Изображения с рамками
```

---

## ⚙️ Установка зависимостей

```bash
# Перейти в папку проекта
cd speedtest

# Установить зависимости
pip install -r requirements.txt

# Или вручную:
pip install ultralytics easyocr opencv-python pandas openpyxl
```

---

## 🗂️ Подготовка датасета

### 1. Конвертация COCO → VOC (если исходные аннотации в COCO)

```bash
# Запустить конвертер
python coco_to_voc.py --input=path/to/coco_annotations --output=path/to/voc_output
```

**Параметры:**
| Аргумент | Описание | Пример |
|----------|----------|--------|
| `--input` | Папка с JSON-аннотациями COCO | `data/coco/annotations` |
| `--output` | Папка для сохранения XML-файлов VOC | `data/voc/annotations` |

**Результат:** В выходной папке появятся `.xml` файлы в формате VOC.

---

### 2. Подготовка датасета для YOLO (локально)

```bash
# Запустить скрипт подготовки
python prepare_dataset.py
```

**Что делает скрипт:**
1. 📁 Сканирует папку `eval/AllFotoAndAnnotations` на наличие изображений и XML-аннотаций
2. 🔄 Конвертирует VOC-аннотации (`.xml`) → YOLO-формат (`.txt`)
3. ✂️ Разделяет данные на train/val/test (70%/20%/10%)
4. 📝 Создаёт `dataset.yaml` с конфигурацией

**Настройки в `prepare_dataset.py`:**
```python
# Пути к исходным данным
SOURCE_IMAGES = "eval/AllFotoAndAnnotations"
SOURCE_ANNOTATIONS = "eval/AllFotoAndAnnotations"

# Выходная папка
OUTPUT_DIR = "data/processed/yolo_speed"

# Пропорции разделения
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Seed для воспроизводимости
SEED = 42
```

**Результат:**
```
data/processed/yolo_speed/
├── dataset.yaml          # Конфигурация для обучения
├── images/
│   ├── train/            # ~165 изображений
│   ├── val/              # ~47 изображений
│   └── test/             # ~25 изображений
└── labels/
    ├── train/            # .txt аннотации
    ├── val/
    └── test/
```

**Пример `dataset.yaml`:**
```yaml
path: F:/Downloads/speedtest-main/speedtest/data/processed/yolo_speed
train: images/train
val: images/val
test: images/test
nc: 3
names: ['Download', 'Upload', 'qms']
```

> ⚠️ **Важно**: Убедитесь, что `nc` (количество классов) и `names` соответствуют вашим аннотациям!

---

## 🎯 Обучение модели

### Команда для обучения (одной строкой):

```bash
yolo detect train data=data/processed/yolo_speed/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 name=qms_final project=runs/detect device=cpu
```

### Параметры обучения:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `data` | `dataset.yaml` | Путь к конфигурации датасета |
| `model` | `yolov8n.pt` | Предобученная модель (n/s/m/l/x) |
| `epochs` | `100` | Количество эпох обучения |
| `imgsz` | `640` | Размер изображения для обучения |
| `batch` | `8` | Размер батча (уменьшите при нехватке памяти) |
| `name` | `qms_final` | Имя эксперимента |
| `project` | `runs/detect` | Папка для сохранения результатов |
| `device` | `cpu` | `cpu` или `0` для GPU |

### 📊 Мониторинг обучения:

- Логи и метрики сохраняются в: `runs/detect/qms_final/`
- Графики: `runs/detect/qms_final/results.png`
- Лучшая модель: `runs/detect/qms_final/weights/best.pt`

> ⏱️ **Время обучения**: ~3-5 часов на CPU, ~30-60 минут на GPU

---

## 🧪 Тестирование модели

### 1. Тест на изображениях QMS (должно быть `qms_detected=True`)

```bash
python main.py eval/by_service/QMS/qms_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
```

### 2. Тест на других сервисах (должно быть `qms_detected=False`)

```bash
# Yandex
python main.py eval/by_service/Yandex/yandex_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# Speedtest.net
python main.py eval/by_service/Speedtest/speedtest_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# Table
python main.py eval/by_service/Table/table_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
```

### 3. Тест на всех изображениях сразу

```bash
# Все фото из одной папки
python main.py eval/AllFotoAndAnnotations --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# Все сервисы из by_service
python main.py eval/by_service --weights="runs/detect/runs/detect/qms_final/weights/best.pt"
```

### 4. Проверка результатов

```bash
# Посмотреть CSV в консоли
type outputs\results.csv

# Посчитать статистику по qms_detected
powershell -Command "Get-Content outputs\results.csv | Select-Object -Skip 1 | ForEach-Object { $_.Split(',')[10] } | Group-Object"
```

**Ожидаемый результат:**
| Сервис | `qms_detected` |
|--------|---------------|
| QMS (qms.ru) | `True` (~95% случаев) |
| Yandex | `False` |
| Speedtest.net | `False` |
| Table | `False` |

---

## 📈 Оценка качества модели (eval.py)

### Запуск оценки:

```bash
# Из папки eval
cd eval
python eval.py --weights=../runs/detect/runs/detect/qms_final/weights/best.pt

# Или полный путь
python F:\Downloads\speedtest-main\speedtest\eval\eval.py --weights=F:\Downloads\speedtest-main\speedtest\runs\detect\runs\detect\qms_final\weights\best.pt
```

### Параметры eval.py:

| Аргумент | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `--weights` | `../weights/best.pt` | Путь к файлу модели |
| `--services_dir` | `by_service` | Папка с тестовыми сервисами |

### 📊 Вывод оценки:

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

**Расшифровка метрик:**
| Метрика | Описание | Хорошее значение |
|---------|----------|-----------------|
| `mAP50` | Средняя точность при IoU=0.5 | > 0.7 |
| `mAP50-95` | Средняя точность при IoU 0.5-0.95 | > 0.5 |
| `Precision` | Доля верных срабатываний | > 0.8 |
| `Recall` | Доля найденных объектов | > 0.8 |

---

## 🔧 Полезные команды терминала

### 🔍 Поиск файлов моделей:

```powershell
# Найти все best.pt в проекте
Get-ChildItem -Path F:\Downloads\speedtest-main -Filter best.pt -Recurse -ErrorAction SilentlyContinue | Select-Object FullName

# Проверить существование файла
Test-Path "runs/detect/runs/detect/qms_final/weights/best.pt"
```

### 📊 Анализ результатов:

```powershell
# Посчитать True/False по qms_detected
Get-Content outputs\results.csv | Select-Object -Skip 1 | ForEach-Object { $_.Split(',')[10] } | Group-Object

# Посчитать среднюю уверенность
Get-Content outputs\results.csv | Select-Object -Skip 1 | ForEach-Object { $_.Split(',')[11] } | Measure-Object -Average

# Найти файлы с qms_detected=False
Select-String -Path outputs\results.csv -Pattern "False," | Select-Object Line
```

### 🗂️ Работа с папками:

```powershell
# Очистить outputs перед новым тестом
Remove-Item outputs\* -Recurse -Force

# Создать резервную копию результатов
Copy-Item outputs\results.csv "backup\results_$(Get-Date -Format 'yyyy-MM-dd_HH-mm').csv"

# Посмотреть структуру папки
tree /F speedtest\eval\by_service
```

### 🐍 Управление окружением:

```bash
# Проверить версию Python
python --version

# Проверить установленные пакеты
pip list | findstr ultralytics
pip list | findstr easyocr

# Обновить ultralytics
pip install -U ultralytics
```

---

## ❓ Частые проблемы и решения

| Проблема | Причина | Решение |
|----------|---------|---------|
| `FileNotFoundError: Weights not found` | Неправильный путь к модели | Использовать полный путь или проверить `runs/detect/runs/detect/` |
| `qms_detected=False` всегда | Модель не обучена на классе qms | Проверить `dataset.yaml`: `nc: 3` и `names: [..., 'qms']` |
| Низкая точность OCR | Плохое качество изображения | Добавить аугментации или использовать `preprocess_variants()` в `ocr.py` |
| Медленная обработка | Работа на CPU | Использовать GPU: `device=0` или уменьшить `batch` |
| Ошибка в eval.py | Несоответствие классов | Синхронизировать `CLASS_NAMES` в `eval.py` с `dataset.yaml` |

---

## 📋 Чеклист перед запуском

### Перед обучением:
- [ ] `dataset.yaml` содержит `nc: 3` и `names: ['Download', 'Upload', 'qms']`
- [ ] В аннотациях есть класс `<name>qms</name>` для QMS-изображений
- [ ] В аннотациях **НЕТ** класса `qms` для не-QMS изображений
- [ ] Папка `data/processed/yolo_speed/` создана скриптом `prepare_dataset.py`

### Перед тестированием:
- [ ] Модель обучена и файл `best.pt` существует
- [ ] Путь к весам указан верно (проверить `runs/detect/runs/detect/`)
- [ ] `detector.py` возвращает класс `"qms"`
- [ ] `main.py` обрабатывает `qms_detected` и `qms_conf`

### Перед оценкой (eval.py):
- [ ] `CLASS_NAMES` в `eval.py` синхронизирован с `dataset.yaml`
- [ ] Папки `by_service/{Service}/{service}_images/` и `_annotations/` существуют
- [ ] Аннотации в формате VOC (`.xml`)

---

## 🚀 Быстрый старт (для новых разработчиков)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd speedtest-main/speedtest

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Подготовить датасет
python prepare_dataset.py

# 4. Обучить модель
yolo detect train data=data/processed/yolo_speed/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 name=qms_final project=runs/detect device=cpu

# 5. Протестировать
python main.py eval/by_service/QMS/qms_images --weights="runs/detect/runs/detect/qms_final/weights/best.pt"

# 6. Оценить качество
cd eval
python eval.py --weights=../runs/detect/runs/detect/qms_final/weights/best.pt
```

---

## 📞 Контакты и поддержка

> **Контакты и поддержка по поводу моих изменений**

| Поле | Значение |
|------|----------|
| **Автор** | Андрей |
| **Telegram** | @mr_mandarin0 |
| **Дата** | Март 2026 |

> 💡 **Совет**: При возникновении ошибок — проверьте пути к файлам и синхронизацию конфигураций (`dataset.yaml`, `eval.py`, `detector.py`).

---

*Документация актуальна для версии проекта от 22.03.2026* 🎯