_base_ = [
    '_base_abcnet_resnet50_fpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_500e.py',
]

# TRUDI images are large (up to ~4k x 4k), and training happens on a single
# GPU, so use a scaled-down RandomChoiceResize compared to the icdar2015 base.
trudi_train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(type='RemoveIgnored'),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(
        type='RandomRotate',
        max_angle=10,
        pad_with_fixed_color=True,
        use_canvas=True),
    dict(
        type='RandomChoiceResize',
        scales=[(720, 1920), (800, 1920), (880, 1920), (960, 1920),
                (1040, 1920), (1080, 1920)],
        keep_ratio=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

trudi_test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1600, 2560), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# dataset settings
trudi_textspotting_train = dict(
    type='OCRDataset',
    data_root='data/trudi',
    ann_file='textspotting_train.json',
    pipeline=trudi_train_pipeline)

trudi_textspotting_test = dict(
    type='OCRDataset',
    data_root='data/trudi',
    ann_file='textspotting_test.json',
    test_mode=True,
    pipeline=trudi_test_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=trudi_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=trudi_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)

find_unused_parameters = True

# finetune: fewer epochs, evaluate more often
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

load_from = 'https://download.openmmlab.com/mmocr/textspotting/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_pretrain-d060636c.pth'  # noqa