import cv2 
from mmengine.registry import init_default_scope
import matplotlib.pyplot as plt 
from mmdet.datasets.coco import CocoDataset
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets.coco_minecraft import CocoMinecraftDataset
import matplotlib.pyplot as plt

# Эта строчка необходима для корректной работы с регистрами mmdetection
# Благодаря регистру mmdetection объекты (модели, backbone, датасеты и т.д.)
# создаются автоматически на основе словарей конфигурации.
init_default_scope('mmdet')


def load_dataset() -> CocoDataset:
    # Относительный путь до корневой папки с данными 
    data_root = 'datasets/minecraft/'

    # CocoDataset умеет работать с форматом COCO 
    # Но какие именно операции будут в пайплайне данных, определяет пользователь
    # Это может быть минимальный набор из 
    # 1. Прочитать данные 
    # 2. Загрузить разметку 
    # Так и куда более сложный пайплайн с аугментациями/фильтрацией данных/и т.д.

    # В нашем случае это будет минимальный набор для визуализации разметки 
    reading_labels_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs')
    ]
    dataset=CocoMinecraftDataset(
        data_root=data_root,
        ann_file="annotations/train_annotations.json",
        data_prefix=dict(img="train"),
        pipeline=reading_labels_pipeline,
    )
    return dataset


def visialize_sample() -> None:
    # Считаем датасет 
    dataset: CocoDataset = load_dataset()

    # Подготовим данные для отрисовки 
    sample = dataset[1]
    print(f"Путь к изображению: {sample['data_samples'].img_path}")
    # Переведем картинку из тензора (CxHxW) к numpy-массиву (HxWxC)
    img = sample['inputs']    
    img = img.permute(1, 2, 0).cpu().numpy()
    # Переведем боксы в формат numpy массивов
    data_sample = sample['data_samples']
    gt_instances = data_sample.gt_instances
    gt_instances.bboxes = gt_instances.bboxes.tensor.numpy()

    # Для отрисовки воспользуемся классом DetLocalVisualizer из mmdetection
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = dataset.metainfo
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=data_sample,
        draw_gt=True,
        draw_pred=False,
        show=False
    )
    viz = visualizer.get_image()
    cv2.imwrite("artifacts/viz.jpg", viz)
    

    # Конвертируем BGR (формат OpenCV/MMCV) в RGB (формат Matplotlib)
    img_rgb = viz[:, :, ::-1]
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")  # Отключаем оси с пикселями
    plt.show()


#visialize_sample()