import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict


def check_ann_files (annotations_dir, files_to_check, exp_classes):
    print("=== Проверка структуры JSON аннотаций ===")

    for file_name in files_to_check:
        file_path = annotations_dir / file_name
        print(f"\nПроверка файла: {file_name}...")

        # 1. Проверяем физическое существование файла
        if not file_path.exists():
            print(f"Файл не найден по пути: {file_path}")
            continue
            
        try:
            # 2. Проверяем синтаксическую корректность JSON (валидность структуры)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"JSON синтаксически корректен (файл успешно прочитан).")
            
            # 3. Проверяем соответствие стандарту COCO (наличие обязательных полей)
            required_keys = ['images', 'annotations', 'categories']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                print(f"Внимание: В структуре COCO отсутствуют ключи: {missing_keys}")
            else:
                print(f"Статистика датасета:")
                print(f"   - Изображений (images): {len(data['images'])}")
                print(f"   - Объектов/BBoxes (annotations): {len(data['annotations'])}")
                print(f"   - Классов (categories): {len(data['categories'])}")
                print(f"   - Ожидается классов ({len(exp_classes)})")
                        
                
                # Выведем список классов из файла для сверки с вашим списком из 17 мобов
                classes_in_file = [cat['name'] for cat in data['categories']]
                print(f"   - Обнаружено классов в файле ({len(classes_in_file)}): {classes_in_file[:5]}...")
                
                
        except json.JSONDecodeError as e:
            print(f"Ошибка в структуре JSON! Файл поврежден.")
            print(f"   Детали ошибки: {e}")
        except Exception as e:
            print(f"Непредвиденная ошибка при чтении файла: {e}")
            
            

def check_img_ann (base_dir, annotations_dir, splits):
    print("=== Проверка соответствия картинок и аннотаций ===")

    for split in splits:
        json_path = annotations_dir / f"{split}_annotations.json"
        img_dir = base_dir / split
        
        if not json_path.exists():
            print(f"\nОшибка: Файл разметки для {split} не найден.")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        print(f"\nСплит [{split.upper()}]:")
        print(f"   - В JSON прописано картинок: {len(images)}")
        print(f"   - Всего объектов (BBoxes): {len(annotations)}")
        
        # 1. Проверяем связь: все ли image_id из аннотаций есть в списке картинок
        json_image_ids = {img['id'] for img in images}
        ann_image_ids = {ann['image_id'] for ann in annotations}
        
        broken_ann_ids = ann_image_ids - json_image_ids
        if broken_ann_ids:
            print(f"   Ошибка: {len(broken_ann_ids)} аннотаций ссылаются на несуществующие ID картинок!")
        else:
            print(f"   Все аннотации корректно привязаны к ID изображений.")
            
        # 2. Проверяем физическое наличие файлов картинок на диске
        missing_files = []
        for img in images:
            file_name = Path(img['file_name']).name
            full_img_path = img_dir / file_name
            
            if not full_img_path.exists():
                missing_files.append(img['file_name'])
                
        if missing_files:
            print(f"   Ошибка: {len(missing_files)} картинок из JSON отсутствуют на диске в папке {split}/")
            print(f"      Пример отсутствующих файлов: {missing_files[:3]}")
        else:
            print(f"   Все файлы изображений физически присутствуют на диске.")




def cls_dispersion(annotations_dir, splits):
    artifacts_dir = Path("./artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)  # Гарантируем создание папки

    print("=== Анализ баланса классов в датасете ===")

    plot_data = []

    for split in splits:
        json_path = annotations_dir / f"{split}_annotations.json"

        if not json_path.exists():
            print(f"\nФайл {split}_annotations.json не найден.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        class_map = {cat["id"]: cat["name"] for cat in categories}

        annotations = data.get("annotations", [])
        class_ids = [ann["category_id"] for ann in annotations]

        counts = Counter(class_ids)
        total_objects = len(class_ids)

        print(
            f"\nРаспределение классов для сплита: {split.upper()} (Всего объектов: {total_objects})"
        )
        print(
            f"{'ID':<5} | {'Название класса':<20} | {'Количество':<10} | {'Доля (%)':<10}"
        )
        print("-" * 55)

        for class_id in sorted(class_map.keys()):
            class_name = class_map[class_id]
            quantity = counts[class_id]
            percentage = (
                (quantity / total_objects * 100) if total_objects > 0 else 0
            )

            print(
                f"{class_id:<5} | {class_name:<20} | {quantity:<10} | {percentage:.2f}%"
            )

            # Сохраняем и количество, и долю для графиков
            plot_data.append(
                {
                    "Класс": class_name,
                    "Количество": quantity,
                    "Доля (%)": percentage,
                    "Сплит": split.upper(),
                }
            )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    # --- Построение графиков через Seaborn ---
    sns.set_theme(style="whitegrid")

    # Создаем холст из двух строк (один график под другим)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))

    # 1. График количества объектов
    sns.barplot(
        data=df, x="Класс", y="Количество", hue="Сплит", palette="muted", ax=axes[0]
    )
    axes[0].set_title(
        "Количество объектов по классам и сплитам", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("")  # Убираем нижнюю подпись у верхнего графика
    axes[0].set_ylabel("Абсолютное количество объектов", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    # 2. График долей в процентах
    sns.barplot(
        data=df, x="Класс", y="Доля (%)", hue="Сплит", palette="muted", ax=axes[1]
    )
    axes[1].set_title(
        "Процентное соотношение классов по сплитам", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Название класса (Mob)", fontsize=11, labelpad=10)
    axes[1].set_ylabel("Доля от всех объектов в сплите (%)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    # Общая корректировка разметки
    plt.tight_layout()

    # Сохраняем сдвоенный график
    output_img = artifacts_dir / "class_distribution_combined.png"
    plt.savefig(output_img, dpi=300)
    plt.show()

    print(f"\n[INFO] Графики распределения сохранены в: {output_img}")



def vis_test (annotations_dir, image_dir):
    # 2. Загрузка аннотаций COCO
    annotations_file = annotations_dir / "test_annotations.json"
    
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    # Создаем словари для быстрого поиска
    images = {img["id"]: img for img in coco_data["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # 3. Выбор тестового изображения и поиск его аннотаций
    # Берем первое доступное изображение из списка
    target_image = coco_data["images"][0]
    target_image_id = target_image["id"]
    image_filename = target_image["file_name"]

    # Собираем все bounding box'ы для этого изображения
    image_annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == target_image_id
    ]

    # 4. Загрузка и отрисовка изображения
    image_path = os.path.join(image_dir, image_filename)
    img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # 5. Отрисовка bounding box'ов и классов
    for ann in image_annotations:
        # COCO формат bbox: [x_min, y_min, width, height]
        bbox = ann["bbox"]
        x, y, w, h = bbox
        
        category_id = ann["category_id"]
        category_name = categories.get(category_id, f"ID {category_id}")
        
        # Создаем рамку (bbox)
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2, 
            edgecolor="red", 
            facecolor="none"
        )
        ax.add_patch(rect)
        
        # Добавляем подпись класса
        ax.text(
            x, y - 5, 
            category_name, 
            color="white", 
            fontsize=12, 
            bbox=dict(facecolor="red", alpha=0.5, pad=2)
        )

    plt.axis("off")
    plt.show()


def img_per_obj(annotations_dir):
    annotations_file = annotations_dir / "test_annotations.json"
    # 2. Загрузка аннотаций COCO
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    # Создаем словарь для соответствия ID категории и её названия
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Словарь, где ключ — ID класса, а значение — множество (set) уникальных image_id
    class_images = defaultdict(set)

    # 3. Собираем уникальные изображения для каждого класса
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        class_images[cat_id].add(img_id)

    # 4. Выводим результаты, отсортированные по убыванию количества картинок
    print(f"{'Класс':<20} | {'Кол-во уникальных картинок':<25}")
    print("-" * 50)

    # Сортируем категории по количеству уникальных картинок
    sorted_classes = sorted(
        class_images.items(), 
        key=lambda item: len(item[1]), 
        reverse=True
    )

    for cat_id, img_ids in sorted_classes:
        class_name = categories.get(cat_id, f"Unknown (ID {cat_id})")
        img_count = len(img_ids)
        print(f"{class_name:<20} | {img_count:<25}")

    # Общая статистика
    total_unique_annotated_images = len(set(ann["image_id"] for ann in coco_data["annotations"]))
    print("-" * 50)
    print(f"Всего изображений в датасете: {len(coco_data['images'])}")
    print(f"Из них имеют хотя бы одну аннотацию: {total_unique_annotated_images}")
