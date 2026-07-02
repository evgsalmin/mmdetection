import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS
import torch
import matplotlib.pyplot as plt

def test_inference_pt():
    torch.manual_seed(42)
    # 1. Указываем пути к файлам
    config_file = "configs/fcos/fcos_minecraft.py"
    checkpoint_file = (
        "checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"
    )
    img_path = "datasets/minecraft/train/snow_-_pig_1215_jpg.rf.004b081d444fa45d89e051a2d793ac9c.jpg"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 2. Инициализируем модель
    model = init_detector(config_file, checkpoint_file, device=device)

    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    # 5. Инициализируем новый визуализатор
    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    coco_dataset = DATASETS.get('CocoDataset')
    
    # Передаем классы и палитру цветов COCO в визуализатор
    visualizer.dataset_meta = {
        'classes': coco_dataset.METAINFO['classes'],
        'palette': coco_dataset.METAINFO['palette']
    }
    # Передаем датасет для автоматической загрузки названий классов (майкрафт-объектов)
    #dataset_cls = DATASETS.get(model.cfg.train_dataloader.dataset.type)
    #custom_classes = list(dataset_cls.METAINFO['classes'])
    #visualizer.dataset_meta = {'classes': custom_classes}
    #print(f"Успешно загружены классы из {model.cfg.train_dataloader.dataset.type}: {custom_classes}")


    # 6. Отрисовываем предсказание
    visualizer.add_datasample(
        "result",
        img,
        data_sample=result,
        draw_gt=False,
        show=False,  # Отключаем показ в отдельном окне OpenCV
        wait_time=0,
        pred_score_thr=0.3,  # Порог уверенности (score threshold)
        out_file="artifacts/inference/test_pretrained.jpg"
    )

    # 7. Отображаем результат прямо в ячейке Jupyter Notebook
    img_vis = visualizer.get_image()
    
    # Конвертируем BGR (формат OpenCV/MMCV) в RGB (формат Matplotlib)
    img_rgb = img_vis[:, :, ::-1]
    mmcv.imwrite(img_vis, "artifacts/inference/test_pretrained.jpg")
    # Отображаем картинку в ячейке Jupyter
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")  # Отключаем оси с пикселями
    plt.show()
