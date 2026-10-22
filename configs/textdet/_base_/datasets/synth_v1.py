synth_v1_textdet_data_root = 'data/synth_v1'

synth_v1_textdet_train = dict(
    type='OCRDataset',
    data_root=synth_v1_textdet_data_root,
    ann_file='train/textdet.json',
    data_prefix=dict(img_path='train'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

synth_v1_textdet_val = dict(
    type='OCRDataset',
    data_root=synth_v1_textdet_data_root,
    ann_file='val/textdet.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)

synth_v1_textdet_test = dict(
    type='OCRDataset',
    data_root=synth_v1_textdet_data_root,
    ann_file='val/textdet.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)
