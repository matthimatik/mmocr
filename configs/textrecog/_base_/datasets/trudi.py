trudi_textrecog_data_root = 'data/trudi'

trudi_textrecog_train = dict(
    type='OCRDataset',
    data_root=trudi_textrecog_data_root,
    ann_file='train/textrecog.json',
    data_prefix=dict(img_path='train'),
    pipeline=None)

trudi_textrecog_val = dict(
    type='OCRDataset',
    data_root=trudi_textrecog_data_root,
    ann_file='val/textrecog.json',
    data_prefix=dict(img_path='val'),
    test_mode=True,
    pipeline=None)

trudi_textrecog_test = dict(
    type='OCRDataset',
    data_root=trudi_textrecog_data_root,
    ann_file='test/textrecog.json',
    data_prefix=dict(img_path='test'),
    test_mode=True,
    pipeline=None)
