_base_ = [
    '../_base_/datasets/synth_v1.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_robustscanner_resnet31.py',
]

# Fine-tune on the synthetic v1 dataset starting from the TRUDI-finetuned
# RobustScanner release weights (configs/textrecog/robust_scanner/README.md).
load_from = ('/data/8hirsch/mmocr/work_dirs/robustscanner_trudi_20e/'
             'robustscanner_resnet31_20e_trudi.pth')

_base_.default_hooks.logger.interval = 50
_base_.default_hooks.checkpoint.interval = 1

synth_v1_train = _base_.synth_v1_textrecog_train
synth_v1_train.pipeline = _base_.train_pipeline

synth_v1_val = _base_.synth_v1_textrecog_val
synth_v1_val.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synth_v1_train)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=synth_v1_val)

test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['SYNTH_V1'])
test_evaluator = val_evaluator

# Fine-tune schedule: same low LR as the TRUDI fine-tune, 10 epochs.
_base_.optim_wrapper.optimizer.lr = 1e-4
param_scheduler = [dict(type='MultiStepLR', milestones=[8, 9], end=10)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
auto_scale_lr = dict(base_batch_size=16)
