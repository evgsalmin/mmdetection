import os
import matplotlib.pyplot as plt
from PIL import Image


def viz_inf_fcos():
# Укажите путь к вашей папке с картинками
    folder_path = 'artifacts/inference/fcos/20260709_172200/images' 
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    # Находим файлы картинок
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    images_to_show = images[:5]

    # Создаем область для 5 графиков в один ряд
    fig, axes = plt.subplots(5, 1, figsize=(20, 20))

    # Отрисовываем каждую картинку в своей ячейке
    for i, img_name in enumerate(images_to_show):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(img_name, fontsize=10) # Заголовок с именем файла
        axes[i].axis('off') # Отключаем оси координат

    # Если картинок в папке оказалось меньше 5, скрываем пустые ячейки
    for j in range(len(images_to_show), 5):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
