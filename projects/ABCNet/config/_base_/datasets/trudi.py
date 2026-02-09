trudi_textspotting_data_root = 'data/trudi'

trudi_textspotting_train = dict(
    type='OCRDataset',
    data_root=trudi_textspotting_data_root,
    ann_file='textspotting_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textspotting_test = dict(
    type='OCRDataset',
    data_root=trudi_textspotting_data_root,
    ann_file='textspotting_test.json',
    test_mode=True,
    # indices=50,
    pipeline=None)
