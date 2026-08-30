_base_ = ['abcnet_resnet50_fpn_500e_trudi_24gbRAM.py']

# Sanity config to confirm the trudi finetuning setup works on 2x 12GB GPUs.
# Inherits the trudi data wiring, the small finetune LR (1e-5 with rec-head
# multipliers) and load_from (SynthText pretrain) from the base config.

_base_.train_dataloader.batch_size = 1
_base_.train_dataloader.num_workers = 2
_base_.val_dataloader.num_workers = 1

_base_.train_cfg.max_epochs = 2
_base_.train_cfg.val_interval = 1
_base_.default_hooks.checkpoint.interval = 1