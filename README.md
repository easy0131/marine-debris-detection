# 해안 쓰레기 탐지 초기 모델

2026 국립공원 위성 모니터링 AI 챌린지 주제 4의 초기 UNet-ResNet18 모델입니다.

- `train_baseline.py`: AIHub 드론 영상으로 미세조정
- `visual_check.py`, `visual_check_template.html`: 로컬 검증 결과 생성 도구
- `submission_initial/`: 대회 제출용 노트북, 의존성, 모델 가중치
- `PLAN.md`: 데이터 처리와 검증 기록

2026-09-18 서버 디버그 제출 점수는 0.2777777778이었습니다. 2026-09-27 추가 학습 모델의 정식 Public 점수는 **0.1800**입니다. 디버그와 Public 점수는 평가 데이터가 달라 직접 비교할 수 없습니다.

## 2026-09-27 실험

- `improve_model.py`: 기존 모델에서 6회 추가 학습, 회전·반전 증강, 임계값 및 반전 추론 비교
- `package_improved.py`: 선택 모델의 제출 노트북 구성과 CSV·RLE 검증
- `experiment_20260927/`: 후보별 점수와 검증 한계를 기록한 `RESULTS.md`, `metrics.json`, `best.json`
- `submission_improved_20260927/`: 정식 제출한 노트북과 의존성. 새 가중치는 Git에 포함하지 않음

기존 형상 점수 구현을 테두리 비교에서 공식 설명에 따른 전체 마스크 비교로 수정했습니다. 과거 metrics 파일은 당시 산식의 기록으로 보존했습니다. 자체 검증 최고점은 0.5500에서 0.5665로 올랐지만 Public 개선은 입증되지 않았습니다. 검증 양성 15장이 원본 타일 3개에 몰려 있고 드론과 위성영상의 차이가 있어, 독립 테스트 성능으로 해석하면 안 됩니다.

스크립트의 데이터 및 제공 원본 모델 경로는 `train_baseline.py`의 `DATA`, `BASE`에 지정되어 있습니다. 실행 전 로컬 경로를 맞추고 루트에 `initial_unet_r18.pt`를 준비한 뒤 `python improve_model.py`, `python package_improved.py` 순서로 실행합니다. 이미 만든 제출 노트북을 사용하려면 해당 폴더의 `assets/model/unet_r18_debris_lite.pt`에 이번 실험의 `best.pt`를 복사해야 합니다.

원본 AIHub 데이터, 영상이 포함된 시각화 결과 HTML, 개인 참여키는 이 저장소에 포함하지 않았습니다. 제출 시 `submission_initial/predict.ipynb`의 마지막 셀에서 본인 참여키를 사용하고, 디버그 검사에는 `--debug`를 붙입니다. 참여키를 넣은 노트북은 저장소에 올리지 마세요.

## 지역 분리 재학습 / Colab

`prepare_coastal.py`는 JSON 클래스명으로 마스크를 복원하고, 겹치는 타일과 30m 완충 구역을 분리하며, 추가 데이터 추출 목록을 생성합니다. `train_spatial.py`는 고정 위치 검증과 GPU 학습을 지원합니다. `make_colab.py`는 데이터·가중치·코드가 포함된 개인용 Colab 묶음을 만듭니다. 자세한 실행 방법과 한계는 [GPU_GUIDE.md](GPU_GUIDE.md)를 보세요.

`package_improved.py --experiment <실험폴더> --template <원본제출폴더> --destination <출력폴더>`로 선택한 실험을 패키징할 수 있습니다. 새 검증 결과를 기존 0.5665 또는 대회 Public 0.1800과 직접 비교하면 안 됩니다.

## 2026-09-30 무료 GPU 환경의 다음 후보

사용자가 확인한 `submission_gpu.zip`의 Public 점수는 **0.3912842453**입니다. 기존 GPU 모델과 CPU 모델의 확률을 같은 비율로 평균하면 추가 학습 없이 자체 검증 점수가 0.4547에서 0.4679로 올랐습니다. 같은 검증셋에서 선택한 탐색 결과이며, 앙상블의 Public 개선은 아직 확인되지 않았습니다. 검증 기록은 `experiment_ensemble_free/verification.json`에 있습니다.

```powershell
python package_improved.py --experiment review_gpu_20260927/experiment --ensemble experiment_combined_cpu --threshold 0.15 --destination submission_ensemble_free
```

이 명령은 두 가중치를 동봉하고 노트북·예측·RLE·ZIP을 검사합니다. 기존 Colab 재학습 절차는 그대로 사용할 수 있으며, 앙상블 패키징에는 추가 GPU 학습이 필요하지 않습니다.

추가 비교에서 같은 두 모델의 임계값 0.20 후보는 **0.4710**이었습니다. `submission_ensemble_t020.zip`은 4개 검증 입력의 노트북 실행, RLE, ZIP 무결성과 가중치 일치를 확인했습니다. 기록은 `experiment_ensemble_free/verification_t020.json`이며 이 후보 역시 Public 점수는 미확인입니다. PC에서 수행한 2회 보완 학습은 이 앙상블을 넘지 못했습니다.

`train_spatial.py --refine`은 기존 가중치를 낮은 학습률로 보완합니다. 학습 음성 영상에서 오탐이 큰 중앙 크롭을 선별해 음성 샘플의 절반에 사용하고, 양성 이미지별 Dice와 CE 가중치 10을 적용합니다. 검증 영상은 오탐 선별에 사용하지 않습니다. `--tta`를 함께 쓰면 시작 가중치도 TTA로 평가하여 기존 점수를 기준점으로 보존합니다.

```powershell
python make_colab.py --data experiment_combined --init review_gpu_20260927/experiment/best.pt --refine --output C:/Users/User/Desktop/해안쓰레기/무료GPU_개선학습
```

위 명령은 같은 비공개 학습 ZIP과 Colab·Kaggle용 노트북을 만듭니다. Kaggle에는 데이터와 노트북을 Private으로 유지합니다. 기본 보완 학습은 12회, 회당 768개 샘플이며 실행 중 최적 가중치와 마지막 체크포인트를 저장합니다. 세션이 삭제되면 로컬 체크포인트도 사라질 수 있으므로 완료 후 Save Version 또는 결과 ZIP 다운로드로 보존합니다.

Kaggle에는 함께 생성되는 `coastal_training_bundle.bin`을 업로드합니다. ZIP과 동일한 내용이며 노트북이 직접 압축을 풉니다. 9월 30일 ZIP 업로드의 서버 처리가 2시간 넘게 지연됐지만, 같은 묶음을 `.bin`으로 올린 데이터는 정상 생성됐습니다.

### Kaggle T4 보완 학습 완료

비공개 [Kaggle Version 2](https://www.kaggle.com/code/jio0131/coastal-debris-free-gpu-refinement)에서 12회 보완 학습을 완료하고 출력 파일을 저장했습니다. 기존과 동일한 검증 크롭 280개에서 6회차 모델(임계값 0.30, TTA 없음)이 선택됐습니다.

| 후보 | 자체 검증 종합 점수 |
| --- | ---: |
| 기존 Public 0.3913 GPU 모델 | 0.454733 |
| 기존 GPU·CPU 앙상블 | 0.471048 |
| **이번 T4 보완 모델** | **0.487589** |
| 이번 모델·기존 CPU 앙상블 최고 | 0.481056 |
| 이번 모델·기존 GPU 앙상블 최고 | 0.482654 |

이번 모델의 패치 macro F1은 0.691282, 형상 F1은 0.283895입니다. 앙상블은 동일 가중치와 임계값 0.10/0.15/0.20/0.25/0.30/0.40/0.50만 비교했습니다. 기록은 `review_kaggle_20260930/comparison.json`에 있습니다. 같은 드론 검증셋을 반복해 선택한 결과이며, 위성영상 Public 성능은 제출 전까지 미확인입니다.

```powershell
python package_improved.py --experiment review_kaggle_20260930/experiment_refine_ce6b62d69b_4e7463e151 --destination submission_kaggle_refined_20260930
```

학습 원본과 체크포인트는 비공개로 보관하고, 제출 파일은 `submission_kaggle_refined_20260930.zip`입니다. Kaggle은 **Save Version에서 출력 저장을 활성화**한 뒤 저장된 버전의 **Output** 탭에서 결과 ZIP을 내려받습니다. 학습이 끝난 GPU 세션은 중지했습니다.
