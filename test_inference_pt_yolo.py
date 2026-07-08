from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def test_inference_pt_yolo():
    torch.manual_seed(42)
    img_path = "datasets/minecraft/train/snow_-_pig_1215_jpg.rf.004b081d444fa45d89e051a2d793ac9c.jpg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yolov8_model = YOLO('yolov8s.pt').to(device)

    img=Image.open(img_path)

    yolov8_output = yolov8_model(img, conf=0.1)[0]

    
    #yolov8_output.save(filename="artifacts/inference/test_pretrained_yolo.jpg")

    annotated_img_bgr = yolov8_output.plot()
    
    cv2.imwrite("artifacts/inference/test_pretrained_yolo.jpg", annotated_img_bgr)
    # Конвертируем в RGB для matplotlib
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_img_rgb)
    plt.axis("off")
    plt.show()
