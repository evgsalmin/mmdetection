import time
import torch
from mmdet.apis import DetInferencer

# 1. Прописываем ваши пути к файлам

def fps_fcos():
    # 1. Прописываем ваши пути к файлам
    CONFIG_FILE = 'configs/fcos/fcos_minecraft.py'
    CHECKPOINT_FILE = 'artifacts/fcos/best_coco_bbox_mAP_epoch_11.pth'
    TEST_IMAGE = "datasets/minecraft/train/snow_-_pig_1215_jpg.rf.004b081d444fa45d89e051a2d793ac9c.jpg"  # Укажите путь к любой вашей картинке Minecraft
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_images = 15
    
    
    # Загружаем инференсер с вашими файлами
    inferencer = DetInferencer(
        model=CONFIG_FILE,
        weights=CHECKPOINT_FILE,
        device=device_name,
        show_progress=False
    )
    
    # Готовим список изображений
    images = [TEST_IMAGE] * num_images
    
    # Прогрев (Warm-up)
    
    for _ in range(5):
        _ = inferencer(TEST_IMAGE)
        
    if 'cuda' in device_name:
        torch.cuda.synchronize()
        
    # Основной замер
    
    start_time = time.time()
    
    for img in images:
        _ = inferencer(img)
        
    if 'cuda' in device_name:
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Расчет результатов
    total_time = end_time - start_time
    fps = round(num_images / total_time, 2)
    avg_ms = (total_time / num_images) * 1000
    
    
    
    return(fps)

