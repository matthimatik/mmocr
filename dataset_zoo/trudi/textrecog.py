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

config_generator = dict(type='TrudiTextRecogConfigGenerator')
