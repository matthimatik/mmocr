_base_ = [
    'abcnet_resnet50_fpn_500e_icdar2015.py',
]

# Either set batch_size=1 or reduce scales of RandomChoiceResize

# _base_.train_dataloader.batch_size = 1

_base_.train_dataloader.dataset.pipeline[5].scales = [
    (830, 2900), (890, 2900), (940, 2900), (1000, 2900),
    (1050, 2900), (1100, 2900), (1160, 2900), (1210, 2900),
    (1270, 2900),
]
