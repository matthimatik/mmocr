trudi_textdet_data_root = 'data/trudi'

trudi_textdet_train = dict(
    type='TrudiDataset',
    split='train',
    data_root=trudi_textdet_data_root,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textdet_val = dict(
    type='TrudiDataset',
    split='val',
    data_root=trudi_textdet_data_root,
    test_mode=True,
    pipeline=None)

trudi_textdet_test = dict(
    type='TrudiDataset',
    split='test',
    data_root=trudi_textdet_data_root,
    test_mode=True,
    pipeline=None)