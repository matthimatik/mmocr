data_root = 'data/trudi'
cache_path = 'data/cache'

repo_url = 'git@github.com:matthimatik/trudi.git'
target = 'v2/text_recognition[type=id_iso6346]/mmocr'

train_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        target=target))

val_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        target=target))

test_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        target=target))

config_generator = dict(
    type='TextRecogConfigGenerator',
    train_anns=[
        dict(ann_file='train/textrecog.json',
             data_prefix=dict(img_path='train'))
    ],
    val_anns=[
        dict(ann_file='val/textrecog.json',
             data_prefix=dict(img_path='val'))
    ],
    test_anns=[
        dict(ann_file='test/textrecog.json',
             data_prefix=dict(img_path='test'))
    ])
