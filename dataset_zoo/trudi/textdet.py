data_root = 'data/trudi'
cache_path = 'data/cache'

repo_url = 'git@github.com:matthimatik/trudi.git'
# Reuse a local TRUDI checkout instead of cloning; set to None to clone
# repo_url into cache_path. `trudi download_images` is idempotent (it skips
# when every image is already present), so keeping it enabled is safe.
local_repo = '/data/8hirsch/trudi'
target = (
    'v2/text_detection[labels=id_text;'
    'type=id_iso6346,id_en13044_1,id_registration_number,'
    'size_and_type_iso6346,size_and_type_adr_rid,size_and_type_mawp,'
    'size_and_type_portable_tank_instruction]/mmocr')

train_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        local_repo=local_repo,
        target=target))

val_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        local_repo=local_repo,
        target=target))

test_preparer = dict(
    obtainer=dict(
        type='TrudiObtainer',
        cache_path=cache_path,
        repo_url=repo_url,
        local_repo=local_repo,
        target=target))

config_generator = dict(
    type='TextDetConfigGenerator',
    train_anns=[
        dict(ann_file='train/textdet.json', data_prefix=dict(img_path='train'))
    ],
    val_anns=[
        dict(ann_file='val/textdet.json', data_prefix=dict(img_path='val'))
    ],
    test_anns=[
        dict(ann_file='test/textdet.json', data_prefix=dict(img_path='test'))
    ])
