import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def video_inf_yolo():
    # 1. Создаем целевую директорию, если она не существует
    output_path = "artifacts/videos/yolo_inference.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Инициализируем модель и видеопоток
    model_yolo = YOLO("artifacts/yolo/weights/best.pt")
    video_path = "datasets/minecraft/video.mp4"  # Укажите путь к исходному видео
    cap = cv2.VideoCapture(video_path)

    # 3. Получаем параметры исходного видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Настраиваем кодек и объект для записи (mp4v отлично работает с .mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Обработка видео запущена...")

    with tqdm(total=total_frames, desc="Обработка видео") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Инференс кадра
            results = model_yolo.predict(frame, conf=0.25, verbose=False)

            # Отрисовка предсказаний
            annotated_frame = results[0].plot()

            # Запись в файл
            out.write(annotated_frame)
            
            # Обновление прогресс-бара
            pbar.update(1)

    # Освобождение ресурсов
    cap.release()
    out.release()
    print(f"\nВидео успешно сохранено в: {output_path}")
