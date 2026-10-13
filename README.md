# 해안 쓰레기 탐지 — Marine Debris Detection

2026 국립공원 위성 모니터링 AI 챌린지 주제 4를 위한 해안 쓰레기 탐지 프로젝트입니다. AIHub 드론 영상으로 UNet-ResNet18을 학습하고, 지역 분리 검증, GPU 학습, 추가 미세조정 및 앙상블을 비교했습니다.

**2026년 10월 3일까지 확인된 정식 제출 중 최고 성능은 `coastal-gpu-v1`의 Public 점수 0.3913입니다.** 해당 제출은 `submission_gpu.zip`에 대응하며, 기존 기록의 원점수는 **0.3912842453**입니다.

## 1. 대회 제출 결과

사용자가 제공한 대회 제출 내역을 기준으로 정리했습니다. 점수는 화면에 표시된 소수점 네 자리이며, 잘린 제출명은 `…`로 유지했습니다.

| 제출일 | 제출명 | Public 점수 |
|---|---|---:|
| 2026-09-27 | `unet-r18-improved-09…` | 0.1800 |
| 2026-09-27 | **`coastal-gpu-v1`** | **0.3913** |
| 2026-09-30 | `coastal-refined-2026…` | 0.3906 |
| 2026-09-30 | `unet-r18-v7-ensemble` | 0.1874 |
| 2026-09-30 | `unet-r18-8tta-postpr…` | 0.1874 |
| 2026-10-01 | `coastal-refined-2026…` | 0.3906 |
| 2026-10-01 | `coastal-ensemble-t02…` | 0.3717 |
| 2026-10-02 | `coastal-segformer-b0…` | 0.3555 |

디버그·연습용 제출은 평가 데이터가 달라 위 표에서 제외했습니다. 오류로 종료되어 점수가 없는 제출도 제외했습니다.

## 2. 주요 모델 비교

저장소의 자체 검증 기록과 대회 제출 결과를 함께 비교하면 다음과 같습니다.

| 후보 | 자체 검증 종합 점수 | Public 점수 | 주요 설정 |
|---|---:|---:|---|
| GPU 단독 모델 | 0.454733 | **0.3913** | 반전 TTA, 임계값 0.10 |
| GPU·CPU 앙상블 | 0.471048 | 0.3717 | 확률 1:1 평균, 임계값 0.20 |
| Kaggle T4 refined 모델 | **0.487589** | 0.3906 | 6회차 체크포인트, 임계값 0.30, TTA 없음 |

자체 검증에서는 refined 모델이 가장 높았지만, Public 평가에서는 GPU 단독 모델이 최고점을 유지했습니다. 앙상블 역시 자체 검증 개선이 Public 개선으로 이어지지 않았습니다.

검증 수치와 설정은 [GPU·CPU 앙상블 검증 기록](https://github.com/easy0131/marine-debris-detection/blob/main/experiment_ensemble_free/verification_t020.json)과 [Kaggle 모델 비교 기록](https://github.com/easy0131/marine-debris-detection/blob/main/review_kaggle_20260930/comparison.json)에 근거합니다. Public 점수는 이후 확인된 제출 결과를 반영했습니다. 해당 JSON의 `null` 또는 “미확인” 문구는 제출 전 기록입니다.

SegFormer-B0와 일부 `unet-r18-*` 후보는 제출 결과만 확인됐으며, 이 README에서는 저장소 내 특정 구현이나 가중치와 연결하지 않습니다.

## 3. 최고 성능 모델: coastal-gpu-v1

현재 Public 평가 기준 모델은 `coastal-gpu-v1`입니다.

- 모델: UNet-ResNet18
- GPU 학습: 30 epochs
- 추론: 반전 TTA
- 예측 임계값: 0.10
- 자체 검증 종합 점수: 0.454733
- Public 점수: 0.3912842453
- 제출 패키지: `submission_gpu.zip`

관련 설정과 성능 검증은 [GPU 실험 폴더](https://github.com/easy0131/marine-debris-detection/tree/main/review_gpu_20260927)에 있습니다.

**현재 저장소에는 이 모델의 학습 가중치와 제출 ZIP이 포함되어 있지 않습니다.** 동일 모델을 실행하려면 별도로 보관한 `submission_gpu.zip` 또는 해당 GPU 실험의 `best.pt`가 필요합니다. 저장소에 있는 초기 모델 가중치는 이 GPU 모델과 구분해야 합니다.

## 4. 데이터 및 검증

AIHub 드론 영상의 JSON 클래스명과 지리 폴리곤을 이용해 쓰레기 마스크를 생성합니다. 클래스 코드의 중복 문제를 피하기 위해 클래스명을 사용하며, 폴리곤 내부 구멍을 보존합니다.

통합 데이터 구성은 다음과 같습니다.

| 구분 | 영상 수 | 양성 영상 수 |
|---|---:|---:|
| 학습 | 661 | 221 |
| 검증 | 70 | 39 |

고정된 해안 지역을 검증에 사용하고, 해당 지역과 인접 30m 구역을 학습에서 제외합니다. 검증 영상마다 네 모서리 패치를 추출해 총 280개 크롭을 평가합니다. 겹치는 패치가 있으므로 모두 독립 표본으로 볼 수는 없습니다.

학습에는 위치·크기 변화, 회전·반전, 밝기·색상·흐림 증강을 사용합니다. 50픽셀 기준은 정답 양성을 제거하는 데 사용하지 않고 예측 마스크에 적용합니다. 상세한 데이터 구성과 준비 절차는 [GPU_GUIDE.md](https://github.com/easy0131/marine-debris-detection/blob/main/GPU_GUIDE.md)에 있습니다.

## 5. 실험 흐름

### 초기 추가 학습

기존 UNet-ResNet18에 6회 추가 학습을 적용하고, 회전·반전 증강과 임계값·반전 추론을 비교했습니다. 이 단계의 정식 제출 점수는 0.1800입니다.

초기 평가 과정에서 형상 점수 구현을 테두리 비교에서 전체 마스크 비교로 수정했습니다. 과거 metrics 파일은 당시 계산 방식의 기록으로 보존되어 있어, 서로 다른 평가 방식의 수치를 직접 비교하면 안 됩니다.

### 지역 분리 및 GPU 학습

고정 검증 지역과 주변 구역을 학습에서 제외하고 학습 데이터를 확장했습니다. GPU에서 30회 학습한 모델을 바탕으로 반전 TTA와 임계값을 선택했으며, `coastal-gpu-v1`이 Public 0.3913을 기록했습니다.

### GPU·CPU 앙상블

기존 GPU 모델과 CPU 모델의 예측 확률을 같은 비율로 평균했습니다. 임계값 0.20에서 자체 검증 점수는 0.471048이었지만, Public 점수는 0.3717로 GPU 단독 모델보다 낮았습니다.

제출 패키지는 검증 입력 4개에 대한 노트북 실행, 예측 일치, RLE 왕복 변환, ZIP 무결성 및 가중치 일치 검사를 통과했습니다. 이는 패키지 동작 확인이며, Public 성능 향상을 의미하지는 않습니다.

### Kaggle T4 보완 학습

기존 GPU 가중치를 낮은 학습률로 추가 학습했습니다. 학습 음성 영상에서 오탐이 큰 크롭을 선별하는 방식과 양성 이미지별 Dice, CE 가중치 조정을 적용했습니다.

12회 보완 학습 중 6회차 모델이 선택됐으며, 임계값 0.30과 TTA 미적용 조건에서 자체 검증 점수 0.487589를 기록했습니다. 이후 Public 점수는 0.3906으로 확인되어 기존 GPU 모델을 넘지는 못했습니다.

## 6. 저장소 구성

| 경로 | 내용 |
|---|---|
| `train_baseline.py` | 초기 UNet-ResNet18 학습 |
| `improve_model.py` | 추가 학습 및 추론 설정 비교 |
| `prepare_coastal.py` | 마스크 생성과 지역 분리 데이터 준비 |
| `train_spatial.py` | 지역 분리 학습, GPU 학습 및 refine |
| `make_colab.py` | Colab·Kaggle 학습 자료 생성 |
| `package_improved.py` | 제출 패키지 생성 및 검증 |
| `visual_check.py` | 예측 결과 시각화 |
| `experiment_20260927/` | 초기 추가 학습 결과 |
| `review_gpu_20260927/` | GPU 모델 설정·성능·제출 코드 |
| `experiment_ensemble_free/` | 앙상블 비교 및 패키지 검증 |
| `review_kaggle_20260930/` | Kaggle 보완 학습 및 모델 비교 |
| `submission_initial/` | 초기 제출 코드와 가중치 |
| `submission_ensemble_t020/` | 임계값 0.20 앙상블 제출 코드 |
| `submission_kaggle_refined_20260930/` | refined 모델 제출 코드 |
| `GPU_GUIDE.md` | 데이터 준비와 GPU 실행 안내 |
| `PLAN.md` | 실험 계획 및 처리 기록 |

## 7. 실행 안내

원본 데이터와 선택한 실험의 가중치를 별도로 준비해야 합니다. 스크립트에 지정된 데이터·모델 경로는 실행 환경에 맞게 수정하세요.

초기 추가 학습과 패키징:

```bash
python improve_model.py
python package_improved.py
```

특정 실험의 제출 패키지 생성:

```bash
python package_improved.py --experiment <실험폴더> --template <원본제출폴더> --destination <출력폴더>
```

GPU·CPU 앙상블 패키지 생성:

```bash
python package_improved.py --experiment review_gpu_20260927/experiment --ensemble experiment_combined_cpu --threshold 0.20 --destination submission_ensemble_t020
```

위 명령은 각 실험에 필요한 가중치와 자료가 준비된 환경을 전제로 합니다. 저장소를 복제하는 것만으로 최고 성능 모델의 추론을 바로 실행할 수는 없습니다.

원본 AIHub 데이터와 개인 참여키는 저장소에 포함하지 않습니다. 참여키를 입력한 제출 노트북은 공개 저장소에 올리지 마세요.

## 8. 결과 해석과 한계

- **Public 평가, 디버그 평가, 자체 검증을 구분합니다.** 데이터와 평가 조건이 달라 점수를 직접 비교할 수 없습니다.
- **자체 검증은 모델 선택용입니다.** 동일한 지역을 반복 사용해 모델과 임계값을 선택했으므로 독립 테스트 결과가 아닙니다.
- **드론과 위성영상의 차이가 있습니다.** 자체 검증 개선이 대회 성능 개선으로 이어지지 않은 원인 후보지만, 각 요인의 영향은 별도 실험으로 확인하지 않았습니다.
- **현재 최고점은 확인된 제출 범위의 결과입니다.** 2026년 10월 3일까지 제공된 정식 제출 내역에서는 `coastal-gpu-v1`이 가장 높은 점수를 기록했습니다.
