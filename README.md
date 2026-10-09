# Executive Deepfake Detection

Code, data splits and results for the paper:

> Ajibawo, O.; Abosata, N.; Bin Sulaiman, R.; Farhan, M. *Defending Against Executive Deepfake Fraud: A Hybrid Detection Framework and Governance Strategy for Enterprise Security.* Standards (MDPI), 2026.

The framework fine-tunes EfficientNetV2-S on face crops, extracts 1,280-dimensional embeddings, and classifies them with an RBF-kernel SVM. A separate SVM classifies 48-dimensional audio features (40 MFCC, 7 spectral contrast, 1 zero-crossing rate). All experiments were run in Google Colab.

## Contents

```
notebooks/   the Colab notebooks, with the outputs from the runs reported in the paper
splits/      partition lists for every dataset
results/     metrics (JSON/CSV) and per-sample decision scores (NPZ) behind every table and figure
```

| Notebook | Covers | Paper |
|---|---|---|
| `01_faceforensics.ipynb` | Official source-video split, face extraction, fine-tuning, features, SVM selection, ablation | Sections 3.2–3.4, 4.2; Tables 1–5 |
| `02_celebdf_and_dfd.ipynb` | Celeb-DF v2 official test list, threshold recalibration, DeepFakeDetection holdout | Section 4.3; Tables 6–8 |
| `03_fakeavceleb.ipynb` | Identity-disjoint split, separate visual and audio labels, fusion strategies, DeLong tests | Section 4.5; Tables 11–12 |
| `04_asvspoof.ipynb` | Official speaker-disjoint partitions, single SVM, per-attack EER | Section 4.4; Tables 9–10 |
| `05_statistics_and_figures.ipynb` | Bootstrap confidence intervals, latency, figure data | Tables 3, 13; Figures 2–3, 6–9 |

## Data

The datasets are not included; their licences do not allow redistribution. Request them from the original authors:

- FaceForensics++ (C23), including the DeepFakeDetection subset: https://github.com/ondyari/FaceForensics
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics
- FakeAVCeleb v1.2: https://github.com/DASH-Lab/FakeAVCeleb
- ASVspoof 2019 Logical Access: https://datashare.ed.ac.uk/handle/10283/3336


- `ffpp_train.json`, `ffpp_val.json`, `ffpp_test.json`: the official FaceForensics++ split (720/140/140 source videos).
- `ffpp_manifest.csv`: every FaceForensics++ video with its partition and source-video group; the DeepFakeDetection subset is marked `dfd_holdout`.
- `celebdf_manifest.csv`: the 518 videos of the official Celeb-DF v2 testing list.
- `fakeavceleb_manifest.csv`: the 1,000 sampled clips, partitioned by identity, with separate visual and audio labels.

ASVspoof 2019 uses its official protocol files unchanged.

## Environment

```
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
```

Training used a single NVIDIA L4 GPU. Latency was measured on one CPU thread (Intel Xeon at 2.20 GHz, 52 GB memory). The random seed is 42 throughout.

## Notes on the results

- The visual backbone was trained once with seed 42; variation across initialisations was not measured.
- `results/ffpp_indistribution.json` is from the first hyperparameter selection, which used cross-validation on the training partition. It was superseded by selection on the validation partition (`ffpp_indistribution_final.json`); see Section 3.4.
- The frozen-backbone ablation (`ffpp_frozen_ablation.json`) used a reduced grid and 30 % of the training groups, so its comparison with the fine-tuned model is an upper bound.
- The DeepFakeDetection subset contains only manipulated videos; its negative class is the real videos of the FaceForensics++ test partition.
- The ASVspoof results come from a single SVM trained on the full training partition.

## Licence

MIT. See `LICENSE`.
