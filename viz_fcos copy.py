import json
import matplotlib.pyplot as plt
import pandas as pd

def viz_fcos():

    log_path = "artifacts/fcos/20260706_135651/vis_data/20260706_135651.json"

    log_data = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                log_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(log_data)

    # 1. Автоматический поиск ключа mAP среди всех колонок
    #map_candidates = ["coco/bbox_mAP", "bbox_mAP", "coco/bbox_mAP_50", "bbox_mAP_50"]


    # 2. Фильтруем данные: берем только те строки, где физически есть нужные метрики
    df_train = df[df["loss"].notna()].copy()
    df_val = df[df["coco/bbox_mAP_50"].notna()].copy()


    # Вывод отладочной информации в консоль
    print(f"Всего строк в логе: {len(df)}")
    print(f"Строк обучения (с loss): {len(df_train)}")
    print(f"Строк валидации (с coco/bbox_mAP_50): {len(df_val)}")

    # 3. Построение графиков
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # График 1: Функция потерь (Loss)
    if not df_train.empty:
        axes[0].plot(
            df_train["step"], df_train["loss"], label="Total Loss", color="crimson"
        )
        if "loss_cls" in df_train.columns:
            axes[0].plot(
                df_train["step"],
                df_train["loss_cls"],
                label="Loss Cls",
                linestyle="--",
            )
        if "loss_bbox" in df_train.columns:
            axes[0].plot(
                df_train["step"],
                df_train["loss_bbox"],
                label="Loss Bbox",
                linestyle="--",
            )
        axes[0].set_title("FCOS Training Losses")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Нет данных loss", ha="center", va="center")

    # График 2: Метрика качества (mAP)
    if not df_val.empty:
        # Используем 'step' для оси X, так как он гарантированно сопоставим с логами обучения
        x_val = df_val["step"] if "step" in df_val.columns else df_val["epoch"]

        axes[1].plot(
            x_val,
            df_val["coco/bbox_mAP_50"],
            marker="o",
            linewidth=2,
            color="dodgerblue",
            label="coco/bbox_mAP_50",
        )
        axes[1].set_title(f"Validation coco/bbox_mAP_50")
        axes[1].set_xlabel("Step / Epoch")
        axes[1].set_ylabel("mAP")

        # Корректный масштаб для маленьких значений (0.04)
        axes[1].set_ylim(-0.05, .5)

        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()
    else:
        # Если df_val пустой, пишем причину на графике
        msg = (
            f"Данные mAP не найдены.\n"
            f"Доступные колонки:\n{list(df.columns)[:5]}... и др."
        )
        axes[1].text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()
