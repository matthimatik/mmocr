_base_ = [
    '_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/synth_v1.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# Fine-tune on the synthetic v1 dataset starting from the TRUDI-finetuned
# DBNet++ release weights (configs/textdet/dbnetpp/README.md).
load_from = ('/data/8hirsch/mmocr/work_dirs/dbnetpp_trudi_200e/'
             'dbnetpp_resnet50-dcnv2_fpnc_200e_trudi.pth')

synth_train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True,
    ),
    dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

synth_v1_textdet_train = _base_.synth_v1_textdet_train
synth_v1_textdet_train.pipeline = synth_train_pipeline

synth_v1_textdet_val = _base_.synth_v1_textdet_val
synth_v1_textdet_val.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synth_v1_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=synth_v1_textdet_val)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

_base_.optim_wrapper.optimizer.lr = 0.0035
_base_.param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=30),
]

_base_.env_cfg.cudnn_benchmark = True

_base_.train_cfg.max_epochs = 30
_base_.train_cfg.val_interval = 5

_base_.default_hooks.checkpoint = dict(type='CheckpointHook', interval=5)

_base_.default_hooks.visualization = dict(
    type='VisualizationHook',
    interval=5,
    enable=True,
    show=False,
    draw_gt=True,
    draw_pred=True)
