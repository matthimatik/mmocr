trudi_textdet_data_root = 'data/trudi'

trudi_textdet_train = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root,
    ann_file='train/textdet.json',
    data_prefix=dict(img_path='train'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textdet_val = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root,
    ann_file='val/textdet.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)

trudi_textdet_test = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root,
    ann_file='test/textdet.json',
    data_prefix=dict(img_path='test'),
    test_mode=True,
    pipeline=None)
