trudi_textspotting_data_root = 'data/trudi'

trudi_textspotting_train = dict(
    type='OCRDataset',
    data_root=trudi_textspotting_data_root,
    ann_file='train/textspotting.json',
    data_prefix=dict(img_path='train'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textspotting_val = dict(
    type='OCRDataset',
    data_root=trudi_textspotting_data_root,
    ann_file='val/textspotting.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)

trudi_textspotting_test = dict(
    type='OCRDataset',
    data_root=trudi_textspotting_data_root,
    ann_file='test/textspotting.json',
    data_prefix=dict(img_path='test'),
    test_mode=True,
    pipeline=None)
