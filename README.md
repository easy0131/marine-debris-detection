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
