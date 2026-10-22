# RobustScanner

> [RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/abs/2007.07542)

<!-- [ALGORITHM] -->

## Abstract

The attention-based encoder-decoder framework has recently achieved impressive results for scene text recognition, and many variants have emerged with improvements in recognition quality. However, it performs poorly on contextless texts (e.g., random character sequences) which is unacceptable in most of real application scenarios. In this paper, we first deeply investigate the decoding process of the decoder. We empirically find that a representative character-level sequence decoder utilizes not only context information but also positional information. Contextual information, which the existing approaches heavily rely on, causes the problem of attention drift. To suppress such side-effect, we propose a novel position enhancement branch, and dynamically fuse its outputs with those of the decoder attention module for scene text recognition. Specifically, it contains a position aware module to enable the encoder to output feature vectors encoding their own spatial positions, and an attention module to estimate glimpses using the positional clue (i.e., the current decoding time step) only. The dynamic fusion is conducted for more robust feature via an element-wise gate mechanism. Theoretically, our proposed method, dubbed \\emph{RobustScanner}, decodes individual characters with dynamic ratio between context and positional clues, and utilizes more positional ones when the decoding sequences with scarce context, and thus is robust and practical. Empirically, it has achieved new state-of-the-art results on popular regular and irregular text recognition benchmarks while without much performance drop on contextless benchmarks, validating its robustness in both contextual and contextless application scenarios.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798010-eee8795e-8cda-4a7f-a81d-ff9c94af58dc.png"/>
</div>

## Dataset

### Train Dataset

|  trainset  | instance_num | repeat_num |           source           |
| :--------: | :----------: | :--------: | :------------------------: |
| icdar_2011 |     3567     |     20     |            real            |
| icdar_2013 |     848      |     20     |            real            |
| icdar2015  |     4468     |     20     |            real            |
| coco_text  |    42142     |     20     |            real            |
|   IIIT5K   |     2000     |     20     |            real            |
| SynthText  |   2400000    |     1      |           synth            |
|  SynthAdd  |   1216889    |     1      | synth, 1.6m in [\[1\]](#1) |
|   Syn90k   |   2400000    |     1      |           synth            |

### Test Dataset

| testset | instance_num |             type              |
| :-----: | :----------: | :---------------------------: |
| IIIT5K  |     3000     |            regular            |
|   SVT   |     647      |            regular            |
|  IC13   |     1015     |            regular            |
|  IC15   |     2077     |           irregular           |
|  SVTP   |     645      | irregular, 639 in [\[1\]](#1) |
|  CT80   |     288      |           irregular           |

## Results and Models

|                               Methods                                | GPUs |        | Regular Text |           |     |           | Irregular Text |        |                               download                                |
| :------------------------------------------------------------------: | :--: | :----: | :----------: | :-------: | :-: | :-------: | :------------: | :----: | :-------------------------------------------------------------------: |
|                                                                      |      | IIIT5K |     SVT      | IC13-1015 |     | IC15-2077 |      SVTP      |  CT80  |                                                                       |
| [RobustScanner](/configs/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real.py) |  4   | 0.9510 |    0.9011    |  0.9320   |     |  0.7578   |     0.8078     | 0.8750 | [model](https://download.openmmlab.com/mmocr/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real_20220915_152447-7fc35929.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real/20220915_152447.log) |
| [RobustScanner-TTA](/configs/textrecog/robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real.py) |  4   | 0.9487 |    0.9011    |  0.9261   |     |  0.7805   |     0.8124     | 0.8819 |                                                                       |

### TRUDI

Fine-tuned on the [TRUDI](https://github.com/matthimatik/trudi) `id_text`
recognition task (all id_text types: `id_iso6346`, `id_en13044_1`,
`id_registration_number`, `size_and_type_iso6346`, `size_and_type_adr_rid`,
`size_and_type_mawp`, `size_and_type_portable_tank_instruction`), starting from
the ICDAR2015 RobustScanner checkpoint above. Dataset: 6025/1263/1219
train/val/test deskewed word crops (legible texts only — `#`-marked and
illegible annotations are excluded by the builder); the transcriptions are
space-free, and the used charset (A-Z, 0-9, `-`) fits the stock
`english_digits_symbols.txt` dictionary. Metrics are word accuracy
(`WordMetric`, exact) reported by `tools/test.py`.

|                                       Method                                       |    BackBone    |               Pretrained Model               | Training set | Test set | #epochs | word_acc |                                     Download                                      |
| :-------------------------------------------------------------------------------: | :------------: | :------------------------------------------: | :----------: | :------: | :-----: | :------: | :------------------------------------------------------------------------------: |
| [RobustScanner](/configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py) |    ResNet31    | [ICDAR2015 RobustScanner](#results-and-models) | TRUDI Train  | TRUDI Val |   20    |  0.7435  | `work_dirs/robustscanner_trudi_20e/robustscanner_resnet31_20e_trudi.pth` (epoch 17) |
| [RobustScanner](/configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py) |    ResNet31    | [ICDAR2015 RobustScanner](#results-and-models) | TRUDI Train  | TRUDI Test |   20    |  0.7071  | `work_dirs/robustscanner_trudi_20e/robustscanner_resnet31_20e_trudi.pth` (epoch 17) |
| [RobustScanner](/configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py) (baseline, before fine-tune) | ResNet31 | ICDAR2015 RobustScanner |     —        | TRUDI Test |    —    |  0.2338  |                                                                                  |

Reproduce (single GPU):

```bash
python tools/dataset_converters/prepare_dataset.py trudi --task textrecog
python tools/train.py configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py \
    --work-dir work_dirs/robustscanner_trudi_20e
# test-split evaluation of the best checkpoint (epoch 17)
python tools/test.py configs/textrecog/robust_scanner/robustscanner_resnet31_20e_trudi.py \
    work_dirs/robustscanner_trudi_20e/epoch_17.pth \
    --cfg-options test_dataloader.dataset.ann_file=test/textrecog.json \
                  test_dataloader.dataset.data_prefix.img_path=test
# package a checkpoint as a release-style weight (drops optimizer state)
python - <<'EOF'
import torch
ckpt = torch.load('work_dirs/robustscanner_trudi_20e/epoch_17.pth', map_location='cpu')
torch.save({k: ckpt[k] for k in ('meta', 'state_dict', 'message_hub', 'param_schedulers')},
           'work_dirs/robustscanner_trudi_20e/robustscanner_resnet31_20e_trudi.pth')
EOF
```

The released `.pth` uses the same layout as the benchmark checkpoints
(`meta` / `state_dict` / `message_hub` / `param_schedulers`), so it loads anywhere an
official MMOCR weight does (e.g. as `load_from` or `--rec-weights`). `train_robustscanner_trudi.sh`
runs the same training command.

### Synthetic v1

Fine-tuned on the synthetic v1 dataset (CTA2024 renders from
`synthetic-data-generation`, ISO6346 container IDs and size/type codes), starting
from the TRUDI checkpoint above. Prepared with
`tools/dataset_converters/cta2024_to_mmocr.py` (scene-level 90/10 train/val
split; legible, horizontal-text instances only, space-free transcriptions):
47309/5714 train/val crops. Metrics are word accuracy (`WordMetric`, exact)
and character precision/recall on the synth val split.

|                                       Method                                       |  BackBone  |               Pretrained Model               | Training set  |  Test set   | #epochs | word_acc |                                     Download                                      |
| :-------------------------------------------------------------------------------: | :--------: | :------------------------------------------: | :-----------: | :---------: | :-----: | :------: | :------------------------------------------------------------------------------: |
| [RobustScanner](/configs/textrecog/robust_scanner/robustscanner_resnet31_10e_synth_v1.py) |  ResNet31  | [TRUDI RobustScanner](#trudi) | Synth v1 Train | Synth v1 Val |   10    |  0.7037  | `work_dirs/robustscanner_synth_v1_10e/robustscanner_resnet31_10e_synth_v1.pth` (epoch 10) |

This is a pure synthetic-domain adaptation: the weights target synthetic imagery
only and are expected to perform poorly on real TRUDI crops (the lower LR than
the detection fine-tune limits, but does not prevent, forgetting of the real
domain).

Reproduce (single GPU):

```bash
# build the synth dataset (hard-links rgb.jpg, writes crops + JSONs)
python tools/dataset_converters/cta2024_to_mmocr.py \
    --src /data/8hirsch/synthetic-data-generation/src/output/dataset \
    --dst data/synth_v1
python tools/train.py configs/textrecog/robust_scanner/robustscanner_resnet31_10e_synth_v1.py \
    --work-dir work_dirs/robustscanner_synth_v1_10e
# package the best checkpoint as a release-style weight (drops optimizer state)
python - <<'EOF'
import torch
ckpt = torch.load('work_dirs/robustscanner_synth_v1_10e/epoch_10.pth', map_location='cpu')
torch.save({k: ckpt[k] for k in ('meta', 'state_dict', 'message_hub', 'param_schedulers')},
           'work_dirs/robustscanner_synth_v1_10e/robustscanner_resnet31_10e_synth_v1.pth')
EOF
```

`train_robustscanner_synth_v1.sh` runs the same training command.

## References

<a id="1">\[1\]</a> Li, Hui and Wang, Peng and Shen, Chunhua and Zhang, Guyu. Show, attend and read: A simple and strong baseline for irregular text recognition. In AAAI 2019.

## Citation

```bibtex
@inproceedings{yue2020robustscanner,
  title={RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition},
  author={Yue, Xiaoyu and Kuang, Zhanghui and Lin, Chenhao and Sun, Hongbin and Zhang, Wayne},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
