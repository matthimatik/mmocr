synth_v1_textrecog_data_root = 'data/synth_v1'

synth_v1_textrecog_train = dict(
    type='OCRDataset',
    data_root=synth_v1_textrecog_data_root,
    ann_file='train/textrecog.json',
    data_prefix=dict(img_path='train'),
    pipeline=None)

synth_v1_textrecog_val = dict(
    type='OCRDataset',
    data_root=synth_v1_textrecog_data_root,
    ann_file='val/textrecog.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)

synth_v1_textrecog_test = dict(
    type='OCRDataset',
    data_root=synth_v1_textrecog_data_root,
    ann_file='val/textrecog.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)
