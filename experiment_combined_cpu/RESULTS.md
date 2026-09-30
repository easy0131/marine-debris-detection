# Training 영상 추가 후 실험

`TS_01.Drone.zip`의 정상 다운로드를 확인했다(41,839,357,662 bytes, TIFF 16,000개). 필요한 영상만 ZIP에서 읽었으며 읽은 멤버의 CRC 검사를 통과했다. 전체 ZIP 모든 멤버의 CRC 검사를 수행한 것은 아니다.

## 데이터

Training과 기존 Validation을 합쳐 학습 661장(양성 221, 음성 440), 검증 70장(양성 39, 음성 31)을 구성했다. 새 Training에서 실제 반영된 영상은 양성 144장과 음성 308장이다. 선별한 Training 영상 중 무효 영역이 있는 8장은 제외했다(양성 2장, 음성 6장).

학습 양성은 이전 77장에서 221장으로 늘었다. 기존 자료의 음성도 통합 후보에서 다시 선별했으므로 이번 실험은 단순히 영상을 덧붙인 구성은 아니다. 데이터 해시와 전체 구성은 `../experiment_combined/data_report.json`, 선별 목록은 `../experiment_combined/extraction_manifest.csv`에 있다.

검증 영상 70개의 ID, RGB 화소, 정답 마스크, 지리 범위가 이전 `experiment_v2/data.npz`와 모두 동일함을 확인했다. 고정된 네 모서리 크롭 280개로 평가하며 양성 크롭은 114개다. 학습과 검증 사이의 지리적 겹침을 배제했다. 크롭과 촬영 날짜가 서로 독립은 아니며, 이 검증 지역에서 모델을 선택하므로 독립 테스트 성능이 아니다.

## 동일 검증셋 결과

| 모델 | 임계값 | macro F1 | 형상 F1 | 종합 |
|---|---:|---:|---:|---:|
| 이전 후보, 시작 가중치 | 0.50 | 0.6426 | 0.1681 | 0.4053 |
| 추가 학습 1회, 선택 | 0.25 | 0.7057 | 0.1500 | **0.4279** |
| 추가 학습 2회 | 0.75 | 0.6639 | 0.1394 | 0.4016 |
| 추가 학습 3회 | 0.40 | 0.6942 | 0.1504 | 0.4223 |
| 추가 학습 4회 | 0.40 | 0.6810 | 0.1721 | 0.4266 |

선택 모델의 TN/FP/FN/TP는 141/25/51/63이다. 이전 후보의 105/61/38/76과 비교하면 오탐은 줄고 놓치는 양성은 늘었다. 종합 점수는 개선됐지만 형상 점수는 낮아졌다. 1회와 4회 모델의 차이는 작으며, 선택 모델의 우위를 대회 데이터까지 일반화할 수 없다.

**대회 서버에 제출하지 않았으며 Public 0.1800의 개선은 아직 확인되지 않았다.**

## 학습·재현

기존 지역 분리 후보 `experiment_spatial_v2/best.pt`에서 시작했다. CPU 4스레드, seed 20260927, 4회, 회당 균형 샘플 256개, batch 8이다. 증강, CE [1,30] + Dice, encoder/decoder 학습률과 워밍업은 `train_spatial.py`를 사용했다. 전체 설정은 `config.json`, 후보별 결과는 `metrics.json`에 저장했다.

```powershell
python train_spatial.py --data experiment_combined/data.npz --init experiment_spatial_v2/best.pt --out experiment_combined_cpu_rerun --epochs 4 --samples 256 --batch-size 8 --device cpu
python package_improved.py --experiment experiment_combined_cpu_rerun --destination submission_combined_rerun
```

## Colab

바탕화면 `해안쓰레기/GPU학습`의 학습 ZIP과 노트북을 새 데이터로 갱신했다. ZIP은 약 235MB다. Colab 기본 설정은 제공 원본 가중치에서 30회 GPU 학습, 회당 512개, batch 16, 반전 추론 비교다. 위 CPU 추가 학습과 초기 가중치 및 학습량이 다르다. 실제 GPU 학습은 아직 실행하지 않았다.

이미 예전 묶음을 업로드했다면 새 ZIP과 노트북을 사용해야 한다. 같은 세션의 기존 파일명·작업 폴더와 무관하게 새로 선택한 업로드 내용을 읽도록 수정했고, 중복 파일명과 ZIP 경로 이탈 회귀 검사를 통과했다.
