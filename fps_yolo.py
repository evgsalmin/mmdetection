import time
import torch
from ultralytics import YOLO

def fps_yolo():
    # 1. Прописываем пути к файлам
    MODEL_FILE = "artifacts/yolo/weights/best.pt"  # Путь к вашей весовой модели YOLO (.pt)
    TEST_IMAGE = "datasets/minecraft/train/snow_-_pig_1215_jpg.rf.004b081d444fa45d89e051a2d793ac9c.jpg"
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_images = 15
    
    # Загружаем модель YOLO
    model = YOLO(MODEL_FILE).to(device_name)
    
    # Готовим список изображений
    images = [TEST_IMAGE] * num_images
    
    # Прогрев (Warm-up)
    for _ in range(5):
        # verbose=False отключает лишний вывод в консоль
        _ = model(TEST_IMAGE, verbose=False)
        
    if 'cuda' in device_name:
        torch.cuda.synchronize()
        
    # Основной замер
    start_time = time.time()
    
    for img in images:
        _ = model(img, verbose=False)
        
    if 'cuda' in device_name:
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Расчет результатов
    total_time = end_time - start_time
    fps = round(num_images / total_time, 2)
    avg_ms = (total_time / num_images) * 1000
    
    return fps
