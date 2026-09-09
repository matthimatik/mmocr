trudi_textspotting_data_root = 'data/trudi'

trudi_textspotting_train = dict(
    type='TrudiDataset',
    split='train',
    data_root=trudi_textspotting_data_root,
    ann_file='textspotting.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textspotting_val = dict(
    type='TrudiDataset',
    split='val',
    data_root=trudi_textspotting_data_root,
    ann_file='textspotting.json',
    test_mode=True,
    pipeline=None)

trudi_textspotting_test = dict(
    type='TrudiDataset',
    split='test',
    data_root=trudi_textspotting_data_root,
    ann_file='textspotting.json',
    test_mode=True,
    pipeline=None)
