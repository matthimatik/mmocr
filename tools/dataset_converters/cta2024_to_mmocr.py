"""Convert CTA2024 text annotations to the MMOCR textdet / textrecog layout.

The synthetic-data-generation (v1) dataset stores one directory per scene with
``rgb.jpg`` and ``cta2024_annotation.jsonl``. This script filters the
annotations down to the ISO6346 container ID codes (CTA2024 ``ID``, e.g.
``ZPGU416912 1``) and the ISO6346 size/type codes (CTA2024 ``COUNTRY_CODE``,
e.g. ``22G1`` / ``45G1``), then writes per-split MMOCR datasets:

* ``<dst>/<split>/textdet.json`` + ``textdet_imgs/`` -- quads for detection,
  scene images hard-linked, fully occluded texts (``percentage_visible`` below
  the ignore threshold) flagged ``ignore``.
* ``<dst>/<split>/textrecog.json`` + ``textrecog_imgs/`` -- tightly cropped
  word images for recognition (legible, horizontal-text instances only),
  transcriptions stripped to whitespace-free text as in the TRUDI builder,
  with the crop rectangle padded to recover edge glyphs clipped by the quad.
  Instances whose quad aspect cannot contain horizontal text (diagonal text
  inside an axis-aligned box) are excluded from recognition.

Scenes are split deterministically by hashing the scene name, so a scene never
appears in both splits.

Example:
    python tools/dataset_converters/cta2024_to_mmocr.py \
        --src /data/8hirsch/synthetic-data-generation/src/output/dataset \
        --dst data/synth_v1
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CTA_TYPES_DEFAULT = ('ID', 'COUNTRY_CODE')

# CTA2024 quads are axis-aligned bounding boxes, not tight text quads: text
# that runs diagonally inside its box (oblique drone views) would produce
# unusable recognition crops, since a single rotation cannot deskew it. A
# horizontal line of N condensed caps has quad aspect ~0.45*N; quads much
# narrower than that cannot contain horizontal text and are dropped from the
# recognition set (they stay in the detection annotations).
RECOG_PER_CHAR_ASPECT = 0.45
RECOG_ASPECT_RATIO = 0.5
RECOG_ASPECT_FLOOR = 1.1


# ---------------------------------------------------------------------------
# geometry / crops
# ---------------------------------------------------------------------------

def parse_corners(corners: str) -> np.ndarray:
    """Parse the ``x1,y1,x2,y2,x3,y3,x4,y4`` corner string into a (4, 2) array."""
    vals = [float(v) for v in corners.split(',')]
    if len(vals) != 8:
        raise ValueError(f'expected 8 corner values, got {len(vals)}')
    return np.array(vals, dtype=float).reshape(4, 2)


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a polygon given as an (n, 2) array."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_bbox(points: np.ndarray) -> list:
    """``[x_min, y_min, x_max, y_max]`` of the point set, as ints."""
    return [
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    ]


def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp vertices into the image, then round to int coordinates."""
    pts = points.copy()
    pts[:, 0] = np.clip(np.round(pts[:, 0]), 0, width)
    pts[:, 1] = np.clip(np.round(pts[:, 1]), 0, height)
    return pts.astype(int)


def text_is_vertical(points: np.ndarray) -> bool:
    """Whether the quad reads vertically (aspect-ratio fallback, as TRUDI)."""
    p0, p1, p2, p3 = points
    height = max(np.linalg.norm(p0 - p3), np.linalg.norm(p1 - p2))
    width = max(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p3))
    return bool(height > width)


def crop_text_region(img: Image.Image,
                     points: np.ndarray,
                     pad_x: float = 0.35,
                     pad_y: float = 0.1) -> Image.Image:
    """Tightly cropped word image for one text quad.

    The CTA2024 quads are axis-aligned and sometimes clip the edge glyphs
    (corner quantisation), so the quad is padded horizontally by ``pad_x``
    and vertically by ``pad_y`` (fractions of the quad height) before the
    crop -- neighbouring text lines are separated vertically, so the extra
    width only picks up wall. The padded quad is masked out of the image
    (everything else white), the crop is rotated to the quad's reading
    direction (identity for the axis-aligned quads) and trimmed to the
    non-white content, mirroring the TRUDI builder
    (``scripts/build/tasks/crop.py``). The mask runs on the bbox crop
    instead of the full image -- equivalent output, far less work on the
    4K source frames.
    """
    p0, p1, p2, p3 = points
    height = float(
        max(np.linalg.norm(p3 - p0), np.linalg.norm(p2 - p1), 1.0))
    xs, ys = points[:, 0], points[:, 1]
    cx, cy = float(xs.mean()), float(ys.mean())
    padded = points.copy()
    padded[:, 0] = np.where(points[:, 0] < cx,
                            xs.min() - pad_x * height,
                            xs.max() + pad_x * height)
    padded[:, 1] = np.where(points[:, 1] < cy,
                            ys.min() - pad_y * height,
                            ys.max() + pad_y * height)

    left = max(int(np.floor(padded[:, 0].min())), 0)
    top = max(int(np.floor(padded[:, 1].min())), 0)
    right = min(int(np.ceil(padded[:, 0].max())) + 1, img.width)
    bottom = min(int(np.ceil(padded[:, 1].max())) + 1, img.height)
    if right - left < 2 or bottom - top < 2:
        return None
    img_crop = img.crop((left, top, right, bottom))

    shifted = np.round(padded - [left, top]).astype(int)
    mask = Image.new('L', img_crop.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in shifted],
                                 outline=1, fill=255)
    masked = Image.new('RGB', img_crop.size, (255, 255, 255))
    masked.paste(img_crop, mask=mask)

    vertical = text_is_vertical(points)
    if vertical:
        m1, m2 = (p0 + p1) / 2, (p2 + p3) / 2
    else:
        m1, m2 = (p0 + p3) / 2, (p1 + p2) / 2
    dx, dy = m2 - m1
    angle = float(np.degrees(np.arctan2(dy, dx)))
    rotated = masked.rotate(
        angle - vertical * 90,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )
    tight = rotated.crop(_surrounding_rect(rotated))
    if tight.size[0] < 2 or tight.size[1] < 1:
        return None
    return tight


def _surrounding_rect(img: Image.Image):
    """Bbox of the non-white content of ``img``."""
    array = np.array(img)
    non_white = np.any(array != [255, 255, 255], axis=-1)
    rows, cols = np.where(non_white)
    if not len(rows) or not len(cols):
        return (0, 0, img.width, img.height)
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1,
            int(rows.max()) + 1)


def strip_transcription(label: str) -> str:
    """Whitespace-free transcription (``"ZPGU416912 1"`` -> ``ZPGU4169121``)."""
    return ''.join(str(label).split())


def quad_aspect(points: np.ndarray) -> float:
    """Width/height ratio of the quad, from the mean of opposite edges."""
    p0, p1, p2, p3 = points
    width = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p3)) / 2
    height = (np.linalg.norm(p3 - p0) + np.linalg.norm(p2 - p1)) / 2
    return float(width / max(height, 1e-6))


def is_horizontally_cropped(points: np.ndarray, text: str) -> bool:
    """Whether the quad plausibly bounds horizontal text of ``text``."""
    min_aspect = max(RECOG_ASPECT_FLOOR,
                     RECOG_ASPECT_RATIO * RECOG_PER_CHAR_ASPECT * len(text))
    return quad_aspect(points) >= min_aspect


# ---------------------------------------------------------------------------
# per-scene worker
# ---------------------------------------------------------------------------

def _is_val_scene(name: str, salt: str, val_ratio: float) -> bool:
    digest = hashlib.md5(f'{salt}:{name}'.encode()).hexdigest()
    return (int(digest, 16) % 10000) < round(val_ratio * 10000)


def process_scene(args):
    """Convert one scene directory; returns (split, det_record, recog_records).

    Recognition crops are written directly by the worker.
    """
    (scene_dir, dst, types, det_min_area, det_ignore_vis, recog_min_vis,
     salt, val_ratio) = args
    scene = scene_dir.name
    ann_path = scene_dir / 'cta2024_annotation.jsonl'
    img_path = scene_dir / 'rgb.jpg'
    if not ann_path.is_file() or not img_path.is_file():
        return None, None, None, f'missing files, skipped: {scene}'

    split = 'val' if _is_val_scene(scene, salt, val_ratio) else 'train'
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    det_instances = []
    recog_records = []
    recog_index = 0
    n_lines = 0
    with ann_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get('text_type') not in types:
                continue
            n_lines += 1
            try:
                points = parse_corners(obj['corners'])
            except ValueError:
                continue
            if polygon_area(points) < det_min_area:
                continue
            vis = float(obj.get('percentage_visible', 0.0))
            pts = clip_points(points, width, height)
            if polygon_area(pts) < det_min_area:
                continue
            det_instances.append({
                'bbox': polygon_bbox(pts),
                'bbox_label': 0,
                'polygon': [int(v) for v in pts.flatten()],
                'text': '*',
                'ignore': vis < det_ignore_vis,
            })
            if vis < recog_min_vis:
                continue
            text = strip_transcription(obj.get('label', ''))
            if not text:
                continue
            if not is_horizontally_cropped(points, text):
                continue
            try:
                crop = crop_text_region(img, points)
            except Exception:
                continue
            if crop is None or crop.width < 16 or crop.height < 10:
                continue
            crop_name = f'{scene}_{recog_index}.jpg'
            recog_index += 1
            crop_path = Path(dst) / split / 'textrecog_imgs' / crop_name
            crop.save(crop_path, format='JPEG', quality=95)
            recog_records.append({
                'img_path': f'textrecog_imgs/{crop_name}',
                'height': crop.height,
                'width': crop.width,
                'instances': [{'text': text, 'ignore': False}],
            })

    det_record = {
        'img_path': f'textdet_imgs/{scene}.jpg',
        'height': height,
        'width': width,
        'instances': det_instances,
    }
    return split, det_record, recog_records, None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def link_scene_image(scene_dir: Path, split_dir: Path) -> None:
    """Hard-link (fall back to copy) the scene image into ``textdet_imgs/``."""
    dst = split_dir / 'textdet_imgs' / f'{scene_dir.name}.jpg'
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(scene_dir / 'rgb.jpg', dst)
    except OSError:
        import shutil
        shutil.copy2(scene_dir / 'rgb.jpg', dst)


def write_json(path: Path, metainfo: dict, data_list: list) -> None:
    with path.open('w', encoding='utf-8') as fh:
        json.dump({'metainfo': metainfo, 'data_list': data_list},
                  fh, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert CTA2024 scene annotations to MMOCR datasets.')
    parser.add_argument(
        '--src',
        default='/data/8hirsch/synthetic-data-generation/src/output/dataset',
        help='Root of the v1 dataset (directories with rgb.jpg + '
        'cta2024_annotation.jsonl).')
    parser.add_argument('--dst',
                        default='data/synth_v1',
                        help='Output dataset root.')
    parser.add_argument('--types',
                        default=','.join(CTA_TYPES_DEFAULT),
                        help='Comma-separated CTA2024 text types to keep.')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--split-salt', default='synth_v1',
                        help='Salt for the deterministic scene split.')
    parser.add_argument('--det-min-area', type=float, default=100.0)
    parser.add_argument('--det-ignore-vis',
                        type=float,
                        default=0.15,
                        help='percentage_visible below this -> ignore=True '
                        'in the detection annotations.')
    parser.add_argument('--recog-min-vis',
                        type=float,
                        default=0.9,
                        help='Minimum percentage_visible for a recognition '
                        'crop.')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    types = tuple(t for t in args.types.split(',') if t)

    scene_dirs = sorted(d for d in src.iterdir()
                        if d.is_dir() and (d / 'rgb.jpg').is_file())
    if not scene_dirs:
        raise SystemExit(f'no scene directories with rgb.jpg found in {src}')

    for split in ('train', 'val'):
        (dst / split / 'textdet_imgs').mkdir(parents=True, exist_ok=True)
        (dst / split / 'textrecog_imgs').mkdir(parents=True, exist_ok=True)

    worker_args = [
        (d, str(dst), types, args.det_min_area, args.det_ignore_vis,
         args.recog_min_vis, args.split_salt, args.val_ratio)
        for d in scene_dirs
    ]

    det_lists = {'train': [], 'val': []}
    recog_lists = {'train': [], 'val': []}
    skipped = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (split, det_record, recog_records,
                warning) in enumerate(pool.map(process_scene, worker_args,
                                               chunksize=8), 1):
            if warning:
                skipped.append(warning)
            if split is None:
                continue
            link_scene_image(
                scene_dirs[i - 1], dst / split)
            det_lists[split].append(det_record)
            recog_lists[split].extend(recog_records)
            if i % 250 == 0:
                print(f'{i}/{len(scene_dirs)} scenes processed')

    det_metainfo = {
        'dataset_type': 'TextDetDataset',
        'task_name': 'textdet',
        'category': [{
            'id': 0,
            'name': 'text'
        }],
    }
    recog_metainfo = {
        'dataset_type': 'TextRecogDataset',
        'task_name': 'textrecog',
    }
    for split in ('train', 'val'):
        write_json(dst / split / 'textdet.json', det_metainfo,
                   det_lists[split])
        write_json(dst / split / 'textrecog.json', recog_metainfo,
                   recog_lists[split])

    print('\nsummary:')
    print(f'  scenes: {len(scene_dirs)} ({len(skipped)} skipped)')
    for w in skipped:
        print(f'    {w}')
    for split in ('train', 'val'):
        n_det = len(det_lists[split])
        n_ign = sum(1 for r in det_lists[split] for inst in r['instances']
                    if inst['ignore'])
        n_tot = sum(len(r['instances']) for r in det_lists[split])
        print(f'  {split}: {n_det} images, {n_tot} det instances '
              f'({n_ign} ignore), {len(recog_lists[split])} recog crops')
    print(f'\nwrote {dst}/{{train,val}}/{{textdet,textrecog}}.json')


if __name__ == '__main__':
    main()
