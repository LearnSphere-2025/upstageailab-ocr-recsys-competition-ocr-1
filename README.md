# OCR Competition Pipeline (DBNet + Hydra)

## Team

| ![김문수](https://avatars.githubusercontent.com/ashrate) | ![이상현](https://avatars.githubusercontent.com/yourshlee) | ![조선미](https://avatars.githubusercontent.com/LearnSphere-2025) | ![채병기](https://avatars.githubusercontent.com/avatar196kc) | ![염창환](https://avatars.githubusercontent.com/cat2oon) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [김문수](https://github.com/ashrate) | [이상현](https://github.com/yourshlee) | [조선미](https://github.com/LearnSphere-2025) | [채병기](https://github.com/avatar196kc) | [염창환](https://github.com/cat2oon) |
| 팀장 · 파이프라인 총괄 | 데이터 증강 · pseudo labeling | 학습 파라미터 & 실험 관리 | 모델 서빙 & 리포트 | 미참여 |

---

## 0. Overview
- **Framework**: PyTorch Lightning 2.x + Hydra 1.2
- **Model**: DBNet (Backbone `ecaresnet50d`, U-Net decoder, DBHead)
- **Environment**: Python 3.10+, CUDA 12.x (선택), Ubuntu 22.04 LTS
- **Dataset**: 제공된 영수증 train/val/test + 외부 공개 데이터(pseudo label)

### Requirements
```bash
pip install -r baseline_code/requirements.txt
```
**주요 의존성**: `pytorch-lightning`, `hydra-core`, `albumentations`, `opencv-python`, `pyclipper`

### Repository Layout (요약)
```
code_1016/                 # 작업 베이스라인 (Lightning + Hydra)
├── configs/               # Hydra 설정 (datasets/models/lightning)
├── runners/               # train / test / predict 엔트리포인트
├── tools/                 # pseudo label & smoothing 유틸
└── outputs/               # 실험 결과 (체크포인트, 로그, 제출물)

code_1017_git/             # 발표 및 배포용 스냅샷
├── checkpoints/           # Top 모델 체크포인트
└── README.md               # 본 문서
```

---

## 1. System Overview
### 학습 파이프라인
1. **Hydra Config** 로더 → `preset=ecaresnet50d` (모델) + `preset/datasets=db_704` (704 스케일 augment)
2. **Lightning Trainer** → 20 epoch, Adam(lr=3e-4) + CosineAnnealingLR
3. **Transforms** → `LongestMaxSize(704)` → `PadIfNeeded(704×704)` → Normalize
4. **Postprocess** → `thresh=0.2`, `box_thresh=0.35`, `max_candidates=600`

### 실행 명령 예시
```bash
# 학습
python code_1016/runners/train.py preset=ecaresnet50d exp_name=ecaresnet50d_scale704_tta

# 검증 / 테스트
python code_1016/runners/test.py \
  preset=ecaresnet50d \
  checkpoint_path=code_1017_git/checkpoints/epoch_epoch=18_step_step=6232.ckpt

# 예측 (수평플립 TTA)
python code_1016/runners/predict.py \
  preset=ecaresnet50d \
  checkpoint_path=code_1017_git/checkpoints/epoch_epoch=18_step_step=6232.ckpt \
  models.submission_dir=outputs/ecaresnet50d_scale704_tta_visual/submissions

# 제출 파일 변환
python code_1016/ocr/utils/convert_submission.py \
  --json_path outputs/.../pred.json \
  --output_path outputs/.../submission.csv
```

---

## 2. Data Pipeline & Pseudo Labeling
### 기본 셋
- `data/datasets/images/{train,val,test}` + `jsons/{train,val}`
- Train 총 3,272장 (val 별도 400장)

### Option 1 – Competition Test 413장
- 모델 추론 → `confidence ≥ 0.85` 박스만 유지
- 결과: 350장 / 1,395박스 → `train_with_test_pseudo.json`
- Hydra preset: `+preset/datasets=db_704_test_pseudo`

### Option 2 – WildReceipt (외부 데이터)
- `.jpeg` 포함 모든 확장자 지원하도록 스크립트 수정
- `confidence ≥ 0.85` 결과: 931장 / 2,865박스 (고신뢰 샘플)

### Option 3 – Cord-v2 & SROIE
- Cord-v2 (1,000장) → 583장 / 1,698박스 @ 0.85
- SROIE Train/Test (556+360장) → 489장 / 1,639박스 @ 0.85

> **참고**: confidence 0.87~0.90 실험도 지원 (`outputs/pseudo_labels/test_high_conf_0{85,87,88,90}.json`).

---

## 3. Experiments
| 실험 | Pseudo Label | 초기화 | Test h-mean |
|---|---|---|---|
| `ecaresnet50d_scale704_tta` | 없음 | ImageNet 사전학습 | **0.9768** |
| `ecaresnet50d_pseudo_full` | test(0.85) from scratch | ImageNet | 0.9737 |
| `ecaresnet50d_pseudo_ft_085` | test(0.85) fine-tune | Top 모델 ckpt | **0.9771** |
| `ecaresnet50d_pseudo_ft_087` | test(0.87) fine-tune | Top 모델 ckpt | **0.9771** |
| `ecaresnet50d_pseudo_full_087` | test(0.87) from scratch | ImageNet | 0.9723 |

- Fine-tuning(`resume=...best.ckpt`)이 from-scratch보다 안정적.
- 0.87 임계값은 데이터가 절반 수준(174장)으로 줄지만, 고신뢰 샘플로 파인튜닝 시 성능 유지.

---

## 4. Leaderboard Snapshot
- **Local Test h-mean**: 0.9768
- **Public Leaderboard h-mean**: 0.9815
- **Private Leaderboard h-mean**: 0.9805
- **Checkpoint**: `code_1017_git/checkpoints/epoch_epoch=18_step_step=6232.ckpt`

---

## 5. Reproducibility Checklist
1. `pip install -r baseline_code/requirements.txt`
2. 데이터 경로 확인 (`data/datasets/**`)
3. 모델 학습 → `train.py preset=ecaresnet50d`
4. 최적 checkpoint 평가 → `test.py`
5. 제출본 생성 → `predict.py` + `convert_submission.py`
6. (선택) pseudo label 적용 시 `tools/generate_pseudo_labels.py` 사용

---

## 6. References
- [DBNet: Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
- [PyTorch Lightning](https://lightning.ai/)
- [Hydra](https://hydra.cc/)
- [Albumentations](https://albumentations.ai/)

---

## 문의
- Issue 또는 PR로 의견을 남겨 주세요.
- 발표 자료 및 로그는 `code_1017_git/` 폴더를 참고하세요.
