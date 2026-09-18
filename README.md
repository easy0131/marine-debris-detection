# 해안 쓰레기 탐지 초기 모델

2026 국립공원 위성 모니터링 AI 챌린지 주제 4의 초기 UNet-ResNet18 모델입니다.

- `train_baseline.py`: AIHub 드론 영상으로 미세조정
- `visual_check.py`, `visual_check_template.html`: 로컬 검증 결과 생성 도구
- `submission_initial/`: 대회 제출용 노트북, 의존성, 모델 가중치
- `PLAN.md`: 데이터 처리와 검증 기록

2026-09-18 서버 디버그 제출은 성공했고 점수는 0.2777777778이었습니다. 디버그 점수는 리더보드에 반영되지 않습니다. 실제 대회 점수는 아직 확인하지 않았습니다.

원본 AIHub 데이터, 영상이 포함된 시각화 결과 HTML, 개인 참여키는 이 저장소에 포함하지 않았습니다. 제출 시 `submission_initial/predict.ipynb`의 마지막 셀에서 본인 참여키를 사용하고, 디버그 검사에는 `--debug`를 붙입니다. 참여키를 넣은 노트북은 저장소에 올리지 마세요.
