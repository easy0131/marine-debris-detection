# 해안쓰레기 GPU 학습

현재 PC는 Intel Arc이며 설치된 PyTorch에서 NVIDIA CUDA를 사용할 수 없습니다. Google Colab의 GPU 런타임을 사용합니다. 무료 GPU 배정·사용시간은 가용량에 따라 달라집니다.

## 실행 순서

1. https://colab.research.google.com/ 에 접속합니다.
2. **파일 → 노트북 업로드**에서 바탕화면 `해안쓰레기/GPU학습/해안쓰레기_GPU학습.ipynb`를 선택합니다.
3. **런타임 → 런타임 유형 변경 → T4 GPU → 저장**을 선택합니다.
4. 셀 왼쪽 실행 버튼을 위에서부터 누릅니다. 업로드 창에서는 같은 폴더의 `coastal_training_bundle.zip` 하나를 선택합니다. ZIP을 먼저 풀 필요는 없습니다.
5. `사용 GPU:` 출력에 GPU 이름이 표시되는지 확인합니다. 데이터 개수도 출력됩니다.
6. 학습 셀은 30 epoch를 실행합니다. 검증 결과에 따라 가중치·임계값을 선택하고 반전 평균 추론도 비교합니다.
7. 다음 셀을 실행하면 노트북·CSV·RLE 검사를 통과한 `submission_gpu.zip`이 다운로드됩니다. **대회 서버에 자동 제출하지 않습니다.**
8. 마지막 선택 셀로 학습 결과를 백업할 수 있습니다. 런타임이 끊기기 전에 다운로드하세요. 같은 런타임의 학습 셀 재실행은 `last.pt`부터 이어갑니다.

무료 GPU 연결이 거절되면 잠시 뒤 다시 시도할 수 있습니다. GPU가 없는 상태로 장시간 CPU 학습을 시작하지 않도록 노트북이 먼저 검사합니다.

## 현재 묶음에 들어 있는 데이터

`data/data_report.json`의 `prepared`가 실제 데이터 구성입니다. 현재 묶음에는 **Training + Validation 드론 영상**을 합쳐 만든 학습 661장(양성 221장), 검증 70장(양성 39장)이 들어 있습니다. 이전 묶음의 학습 양성은 77장이었습니다. 검증 영상·정답·지역은 이전과 동일하게 유지했습니다.

이전 ZIP을 Colab에 이미 올렸다면 **새 `coastal_training_bundle.zip`으로 다시 업로드**해야 합니다. 학습 데이터의 해시가 바뀌어 새 실험으로 실행되므로, 예전 데이터로 학습한 체크포인트와 혼동하지 않습니다.

마스크는 JSON의 `ANN_NM`과 지리 폴리곤으로 만들었습니다. 코드 80의 산림/쓰레기 충돌을 피하고 폴리곤 내부 구멍을 보존합니다. 검증은 고정된 해안 지역이며, 학습에서 이 지역과 인접 30m를 제외합니다. 검증은 영상마다 고정된 네 모서리 패치를 사용하므로 서로 겹치는 표본은 독립이 아닙니다.

정답이 50픽셀보다 작아도 검증 양성에서 제거하지 않습니다. 50픽셀 규칙은 공식 설명대로 **예측 마스크**에 적용합니다. 학습은 다양한 위치·크기의 패치, 회전·반전, 밝기·색·흐림 변화를 사용합니다.

이 점수는 AIHub 드론 자료에서 계산한 모델 선택용 검증값입니다. 위성 집적대에 대한 대회 Public 점수와 다릅니다. 지역 하나를 반복 검증하므로 독립 테스트 성능도 아닙니다.

## 데이터 준비 기록

Training JSON 16,000개 중 쓰레기 양성 영상은 **146개**, 원본 타일은 **40개**입니다. 최초에는 양성 146개와 해안 음성 292개의 우선 추출 목록을 만들었고, 통합 데이터에서는 전체 후보에서 음성을 다시 선별했습니다. 최종 목록은 `experiment_combined/extraction_manifest.csv`이며, 무효 영상 영역이 있는 8장은 제외했습니다.

`Training/01.원천데이터/TS_01.Drone.zip`에서 필요한 영상만 압축 해제 없이 읽었습니다. 사용한 파일들은 ZIP 읽기 과정에서 CRC 검사를 통과했습니다. 전체 ZIP의 모든 영상에 대한 CRC 검사를 수행한 것은 아닙니다. `TL_01.LABEL_01.Drone.zip`은 JSON으로 마스크를 만들기 때문에 필수가 아닙니다.

데이터 통합과 Colab ZIP 생성을 재현하려면 다음 명령을 저장소 폴더에서 실행합니다. 경로는 실제 저장 위치에 맞춥니다.

```powershell
$validation = 'C:\Users\User\Downloads\310.AI기반 국립공원 변화탐지 모니터링 플랫폼 구축\01-1.정식개방데이터\Validation'
$training = 'C:\Users\User\Desktop\해안쓰레기\310.AI기반 국립공원 변화탐지 모니터링 플랫폼 구축\01-1.정식개방데이터\Training'
python prepare_coastal.py --labels "$validation\02.라벨링데이터\VL_02.JSON_01. Drone.zip" --images "$validation\01.원천데이터\VS_01.Drone.zip" --labels "$training\02.라벨링데이터\TL_02.JSON_01.Drone.zip" --images "$training\01.원천데이터\TS_01.Drone.zip" --holdout experiment_v2/data_report.json --out experiment_combined
if ($LASTEXITCODE -eq 0) { python make_colab.py --data experiment_combined }
```

통합 시 음성 선별은 전체 후보에서 다시 진행하므로 최초 438개 추출 목록과 일부 달라질 수 있습니다. ZIP 전체를 보관하는 편이 편리합니다.

공식 안내: https://research.google.com/colaboratory/faq.html

대회 입력·정답: https://aifactory.space/ko/competitions/9307/data

대회 평가: https://aifactory.space/ko/competitions/9307
