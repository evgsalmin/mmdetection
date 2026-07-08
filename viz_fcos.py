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

    # Разделение на train и val
    df_train = df[df["loss"].notna()].copy()
    
    # Ищем любую колонку, содержащую mAP для фильтрации val-строк
    map_cols = [c for c in df.columns if "mAP" in c]
    df_val = df[df[map_cols].notna().any(axis=1)].copy() if map_cols else pd.DataFrame()

    print(f"Всего строк в логе: {len(df)}")
    print(f"Строк обучения (с loss): {len(df_train)}")
    print(f"Строк валидации (с mAP): {len(df_val)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- График 1: ВСЕ компоненты Loss ---
    if not df_train.empty:
        # Основной лосс
        axes[0].plot(df_train["step"], df_train["loss"], label="Total Loss", color="black", linewidth=2)
        
        # Список потенциальных лоссов в FCOS (MMDetection)
        loss_components = {
            "loss_cls": ("Loss Cls (Классификация)", "--"),
            "loss_bbox": ("Loss Bbox (Регрессия рамок)", "--"),
            "loss_centerness": ("Loss Centerness (Центрированность)", ":")
        }
        
        for col, (label, style) in loss_components.items():
            if col in df_train.columns:
                axes[0].plot(df_train["step"], df_train[col], label=label, linestyle=style, alpha=0.8)

        axes[0].set_title("FCOS Training Losses Breakdown")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Нет данных loss", ha="center", va="center")

    # --- График 2: РАСШИРЕННЫЕ метрики mAP ---
    if not df_val.empty:
        x_val = df_val["step"] if "step" in df_val.columns else df_val["epoch"]
        
        # Список стандартных метрик COCO mAP
        map_metrics = {
            "coco/bbox_mAP": ("mAP @[0.5:0.95]", "-"),
            "coco/bbox_mAP_50": ("mAP @0.50", "o-"),
            "coco/bbox_mAP_75": ("mAP @0.75", "--"),
            "coco/bbox_mAP_s": ("mAP (Small)", ":"),
            "coco/bbox_mAP_m": ("mAP (Medium)", ":"),
            "coco/bbox_mAP_l": ("mAP (Large)", ":")
        }
        
        has_plots = False
        for col, (label, style) in map_metrics.items():
            if col in df_val.columns:
                axes[1].plot(x_val, df_val[col], style, label=label, alpha=0.9)
                has_plots = True
                
        # Если точных совпадений нет, выводим все колонки, где есть "mAP"
        if not has_plots:
            for col in map_cols:
                axes[1].plot(x_val, df_val[col], label=col)

        axes[1].set_title("Validation COCO Metrics")
        axes[1].set_xlabel("Step / Epoch")
        axes[1].set_ylabel("Value")
        axes[1].set_ylim(-0.02, 1.0) # mAP обычно от 0 до 1
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()
    else:
        msg = f"Данные mAP не найдены.\nДоступные колонки:\n{list(df.columns)[:5]}..."
        axes[1].text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()
