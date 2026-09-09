trudi_textrecog_data_root = 'data/trudi'

trudi_textrecog_train = dict(
    type='TrudiDataset',
    split='train',
    data_root=trudi_textrecog_data_root,
    ann_file='textrecog.json',
    pipeline=None)

trudi_textrecog_val = dict(
    type='TrudiDataset',
    split='val',
    data_root=trudi_textrecog_data_root,
    ann_file='textrecog.json',
    test_mode=True,
    pipeline=None)

trudi_textrecog_test = dict(
    type='TrudiDataset',
    split='test',
    data_root=trudi_textrecog_data_root,
    ann_file='textrecog.json',
    test_mode=True,
    pipeline=None)
