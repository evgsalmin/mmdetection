import json

def maps_fcos():
    # Путь к вашему файлу
    file_path = "artifacts/inference/fcos/20260709_172200/20260709_172200.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        map_val = data.get("coco/bbox_mAP")
        map_50_val = data.get("coco/bbox_mAP_50")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

    return map_val, map_50_val
