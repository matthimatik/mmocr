_base_ = [
    '_base_abcnet_resnet50_fpn.py',
    '../_base_/datasets/trudi.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_500e.py',
]
# runtime settings
_base_.default_hooks.visualization = dict(
    type='VisualizationHook',
    interval=20,
    enable=True,
    show=False,
    draw_gt=True,
    draw_pred=True,
)

# model settings
_base_.model.roi_head.rec_head.decoder.module_loss.ignore_char = "#"  # "#" is the char for illegible characters in TRUDI
_base_.model.roi_head.rec_head.decoder.max_seq_len = 14
# _base_.model.backbone.frozen_stages = 2  # could be used to boost the recognition head

# schedule settings
_base_.optim_wrapper.optimizer.lr = 0.00001
# recognition head needs more adjustment compared to detection, so we set a higher learning rate multiplier for it
_base_.optim_wrapper.paramwise_cfg=dict(custom_keys={
    'roi_head.rec_head.encoder': dict(lr_mult=2.0),
    'roi_head.rec_head.decoder': dict(lr_mult=5.0),
})
_base_.train_cfg.val_interval = 5
_base_.train_cfg.max_epochs = 200

# dataset settings
trudi_textspotting_train = _base_.trudi_textspotting_train
trudi_textspotting_train.pipeline = _base_.train_pipeline
trudi_textspotting_test = _base_.trudi_textspotting_test
trudi_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    # num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=trudi_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    # num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=trudi_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)

# pretrained on synthtext
load_from = 'https://download.openmmlab.com/mmocr/textspotting/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_pretrain-d060636c.pth'  # noqa

# pretrained on synthtext + icdar2015
# load_from = "https://download.openmmlab.com/mmocr/textspotting/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_500e_icdar2015-326ac6f4.pth"  # noqa

find_unused_parameters = True

train_dataloader["dataset"]["pipeline"][5]["scales"] = [
    (740, 2600), (800, 2600), (860, 2600), (920, 2600),
    (980, 2600), (1040, 2600), (1100, 2600),
]
