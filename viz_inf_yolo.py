import os
import matplotlib.pyplot as plt
from PIL import Image


def viz_inf_yolo():
# Укажите путь к вашей папке с картинками
    image_path = 'artifacts/inference/yolo/val_batch0_pred.jpg' 
    
    img = Image.open(image_path)
        
    
    plt.imshow(img)
    plt.axis('off') # Прячем оси координат (сетку с пикселями)
    plt.tight_layout()
    plt.show() # Отображаем картинку в Jupyter

# ОБЯЗАТЕЛЬНО: вызываем функцию, чтобы она выполнилась
