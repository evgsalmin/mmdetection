import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from tqdm import tqdm
import torch


def video_inf_fcos():

    # --- НАСТРОЙКИ ---
    config_file = 'configs/fcos/fcos_minecraft.py'
    checkpoint_file = 'artifacts/fcos/best_coco_bbox_mAP_epoch_11.pth'
    video_path = 'datasets/minecraft/video.mp4'  # Укажите путь к вашему видео
    output_path = 'artifacts/videos/fcos_inference.mp4'
    score_thr = 0.3  # Порог уверенности для отображения детекций
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
    model_fcos = init_detector(config_file, checkpoint_file, device=device)

    # Настройка визуализатора MMDetection
    visualizer = VISUALIZERS.build(model_fcos.cfg.visualizer)
    visualizer.dataset_meta = model_fcos.dataset_meta

    # --- ОБРАБОТКА ВИДЕО ---
    video_reader = mmcv.VideoReader(video_path)
    video_writer = None

    print(f"Обработка видео: {video_path}...")
    for frame in tqdm(video_reader):
        # Инференс одного кадра
        result = inference_detector(model_fcos, frame)
        
        # Визуализация предсказаний на кадре
        visualizer.add_datasample(
            name='video_frame',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            pred_score_thr=score_thr
        )
        # Получаем кадр с отрисованными боксами (конвертируем из RGB обратно в BGR для OpenCV)
        vis_frame = visualizer.get_image()
        
        #vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Инициализация записи при обработке первого кадра
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                video_reader.fps, 
                (vis_frame.shape[1], vis_frame.shape[0])
            )
            
        video_writer.write(vis_frame)

    # Освобождаем ресурсы
    if video_writer is not None:
        video_writer.release()
    print(f"Готово! Видео сохранено в: {output_path}")
