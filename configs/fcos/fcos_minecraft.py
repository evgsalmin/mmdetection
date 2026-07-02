_base_ = "fcos_r50-caffe_fpn_gn-head_1x_coco.py"

# Модель
# Для адаптации модели к новому датасету нам нужно обновить число классов
model = dict(
    bbox_head=dict(num_classes=18)
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True), # Изменение размера
    dict(type='RandomFlip', prob=0.5),                        # Аугментация: Отражение
    dict(type='PhotoMetricDistortion',                        # Аугментация: Искажение цвета/яркости
         brightness_delta=32, 
         contrast_range=(0.5, 1.5), 
         saturation_range=(0.5, 1.5), 
         hue_delta=18),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),             # Нужно для подсчета метрик на валидации
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


# Датасет 
# train_dataloader и val_dataloader уже определены в конфиге, от которого 
# мы наследуемся

# Переопределяем тип датасета и пути до данных, а остальное используем как есть 
dataset_type = 'CocoMinecraftDataset'
data_root = 'datasets/minecraft/'

train_dataloader = dict(
    batch_size=2,  # Это аналог samples_per_gpu
    num_workers=2, # Это аналог workers_per_gpu
    dataset=dict(
        type='CocoMinecraftDataset',
        data_root=data_root,
        ann_file='annotations/train_annotations.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,  # Для валидации обычно ставят batch_size=1
    num_workers=2, 
    dataset=dict(
        type='CocoMinecraftDataset',
        data_root=data_root,
        ann_file='annotations/valid_annotations.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/test_annotations.json",
        data_prefix=dict(img="test"),
    )
)

# Метрики
# Чтобы корректно посчитать метрики, 
# Нужно переопределить путь до файла с ground true-разметкой 
val_evaluator = dict(
    ann_file=data_root + "annotations/valid_annotations.json",
)
test_evaluator = dict(
    ann_file=data_root + "annotations/test_annotations.json",
)

optim_wrapper = dict(
    #CUDA
    #type='AmpOptimWrapper',                      # Включает AMP (FP16)
    #loss_scale='dynamic',                        # Динамическое масштабирование потерь
    #CPU
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Расписание обучения на 12 эпох (1x schedule) с линейным прогревом (warmup)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=12, milestones=[8, 11], gamma=0.1)
]

# Главный цикл обучения (Runner/Loop)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Настройка логирования и сохранения чекпоинтов (Хуки)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,            # Проверять каждую эпоху
        max_keep_ckpts=3,      # Хранить не более 3 последних обычных чекпоинтов
        save_best='bbox_mAP',  # Автоматически сохранить веса с наилучшим mAP (Исправлено)
        rule='greater'         # Правило: чем больше метрика, тем лучше
    ),
    logger=dict(type='LoggerHook', interval=50)
)

#work_dir = './my_minecraft_experiments/fcos_v1'
