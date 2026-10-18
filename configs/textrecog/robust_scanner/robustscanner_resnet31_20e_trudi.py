_base_ = [
    '../_base_/datasets/trudi.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_robustscanner_resnet31.py',
]

# ICDAR2015-pretrained RobustScanner (openmmlab release layout).
load_from = ('/export/home/8hirsch/.cache/torch/hub/checkpoints/'
             'robustscanner_resnet31_5e_st-sub_mj-sub_sa_real_'
             '20220915_152447-7fc35929.pth')

_base_.default_hooks.logger.interval = 50
_base_.default_hooks.checkpoint.interval = 1

trudi_train = _base_.trudi_textrecog_train
trudi_train.pipeline = _base_.train_pipeline

trudi_val = _base_.trudi_textrecog_val
trudi_val.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=trudi_train)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=trudi_val)

test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['TRUDI'])
test_evaluator = val_evaluator

# Fine-tune schedule: lower LR than the from-scratch 5e recipe, 20 epochs.
_base_.optim_wrapper.optimizer.lr = 1e-4
param_scheduler = [dict(type='MultiStepLR', milestones=[16, 18], end=20)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
auto_scale_lr = dict(base_batch_size=16)
