import os
import matplotlib.pyplot as plt
import pandas as pd

def viz_yolo():
    # 1. Указываем путь к файлу результатов
    csv_path = "artifacts/yolo/results.csv"

    # Проверяем, существует ли файл, чтобы избежать ошибок
    if os.path.exists(csv_path):
        # 2. Читаем данные (удаляя лишние пробелы в названиях колонок)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # 3. Настраиваем сетку графиков (1 строка, 2 колонки)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # График функций потерь (Loss)
        # В зависимости от версии YOLO названия могут чуть отличаться (например, train/box_loss или train/box_obj_loss)
        # Автоматически ищем колонки, содержащие 'loss'
        loss_cols = [col for col in df.columns if "loss" in col.lower()]
        for col in loss_cols:
            axes[0].plot(df["epoch"], df[col], label=col)
        axes[0].set_title("Функции потерь (Loss)")
        axes[0].set_xlabel("Эпоха")
        axes[0].set_ylabel("Значение")
        axes[0].legend()
        axes[0].grid(True)

        # График метрик точности (mAP)
        map_cols = [
            col for col in df.columns if "map" in col.lower() or "metrics/" in col
        ]
        for col in map_cols:
            axes[1].plot(df["epoch"], df[col], label=col)
        axes[1].set_title("Метрики качества (mAP / Precision / Recall)")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Значение")
        axes[1].legend()
        axes[1].grid(True)

        plt.savefig('artifacts/yolo/graph.jpg', dpi=300, bbox_inches="tight")

        plt.tight_layout()
        plt.show()
    else:
        print(f"Файл не найден по пути: {csv_path}")
        print("Убедитесь, что обучение завершилось или уже создало этот файл.")
