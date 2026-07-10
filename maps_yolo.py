import pandas as pd

def maps_yolo():
    # 1. Загружаем файл
    df = pd.read_csv(r"artifacts\inference\yolo\metrics.csv")

    # 2. Очищаем пробелы в названиях колонок (на всякий случай)
    df.columns = df.columns.str.strip()
    df["Metric"] = df["Metric"].str.strip()

    # 3. Извлекаем значения
    map_50 = df.loc[df["Metric"] == "mAP50(B)", "Value"].values[0]
    map_95 = df.loc[df["Metric"] == "mAP50-95(B)", "Value"].values[0]

    return map_95, map_50

