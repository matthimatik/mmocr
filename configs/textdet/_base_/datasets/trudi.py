trudi_textdet_data_root = '/data/8hirsch/trudi/builds/v2/text_detection[type=id_iso6346]/mmocr'

trudi_textdet_train = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root + '/train',
    ann_file='textdet.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

trudi_textdet_val = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root + '/val',
    ann_file='textdet.json',
    test_mode=True,
    pipeline=None)

trudi_textdet_test = dict(
    type='OCRDataset',
    data_root=trudi_textdet_data_root + '/test',
    ann_file='textdet.json',
    test_mode=True,
    pipeline=None)