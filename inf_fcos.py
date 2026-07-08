import os
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from IPython.display import Image, display

def inf_fcos(img_path):
    config_file = 'configs/fcos/fcos_minecraft.py'
    checkpoint_file = 'artifacts/fcos/best_coco_bbox_mAP_epoch_11.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Инициализация модели
    model_fcos = init_detector(config_file, checkpoint_file, device=device)

    visualizer = VISUALIZERS.build(model_fcos.cfg.visualizer)
    visualizer.dataset_meta = model_fcos.dataset_meta

    # 2. Чтение изображения и инференс
    img = mmcv.imread(img_path)
    result = inference_detector(model_fcos, img)

    # 3. Настройка целевой папки и путей
    out_dir = 'artifacts/inference/fcos'
    os.makedirs(out_dir, exist_ok=True) # Создает папку, если её еще нет
    
    # Извлекаем только имя файла (например, 'pic.jpg')
    file_name = os.path.basename(img_path) 
    base, ext = os.path.splitext(file_name)
    
    # Собираем итоговый путь: 'artifacts/inference/fcos/pic_result.jpg'
    out_path = os.path.join(out_dir, f"{base}_result{ext}")

    # 4. Отрисовка
    img_rgb = mmcv.bgr2rgb(img) 
    visualizer.add_datasample(
        name='fcos_result',
        image=img_rgb,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.3,
        show=False,
        out_file=out_path
    )

    # 5. Вывод результата в Jupyter
    display(Image(filename=out_path))
