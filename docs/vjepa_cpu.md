# V-JEPA 2 CPU 시간 특징 비교

상태: 코드 준비, 로컬 인과적 창·평가·재개 검사 및 작은 VJEPA2 모델의 실제 CPU API 검사 완료. 공식 사전학습 체크포인트의 실제 추론·성능은 Colab 실행에서 확인한다.

## 목적 및 범위

R01에서 기존 좌표 기반 움직임과 영상 인코더 특징을 비교한다. `facebook/vjepa2-vitl-fpc64-256`의 사전학습 인코더를 고정해 사용한다. 원래 V-JEPA의 재현이라고 부르지 않으며, 사용 버전은 **V-JEPA 2 ViT-L/16**이다.

첫 CPU 실험은 16개의 실제 연속 과거 프레임으로 구성한 clip을 32프레임 간격으로 평가한다. 전체 49개 영상에 총 349개 구간, 테스트 영상에 113개 평가 시점이 있다. 미래 프레임은 사용하지 않고 마지막 프레임의 라벨로 평가한다. 다른 시점의 점수를 보간하거나 정상으로 대체하지 않는다.

공식 체크포인트는 64프레임·256해상도 설정이다. 이번 예비 실험은 CPU 연산을 줄이기 위해 입력 길이를 16으로 줄였으며 256해상도는 유지했다. 모델의 실제 encoder API가 짧은 입력을 처리하는지 실행 초반에 확인한다. 이러한 입력 설정 차이와 sparse 평가를 결과에 기록한다.

## 입력 및 모델

- 영상별 전체 화면의 동일한 중앙 crop을 시간 순서대로 입력한다. **객체 중심 V-JEPA 실험은 아니다.**
- 이미지 shortest-side 256 / center-crop 256 / bicubic / ImageNet 정규화를 사용한다. 공간 crop 비율은 이전 DINO의 224 설정과 맞추되 해상도는 다르다. 공식 processor의 shortest-side 292 설정과 차이가 있다.
- 예측기 및 action classifier는 이상 판정에 사용하지 않는다. 마지막 encoder token들을 평균하고 L2 정규화해 1024차원 특징을 만든다.
- 초기에는 약 1.3 GB의 공개 가중치 파일을 다운로드한다. 기존 검출·추적과 DINOv2 캐시를 재사용하므로 이 모델들은 다시 실행하지 않는다.
- 첫 실제 clip의 처리 시간으로 약식 전체 시간을 추정한다. 예측 시간은 다운로드·정상 보정·그래프 출력·속도 변화를 포함하지 않는다.
- CPU 추론은 한 clip씩 수행한다. 미세 조정이나 역전파는 없다.

## 비교하는 방법

| 코드 | 특징 |
|---|---|
| G | 마지막 프레임의 저장된 DINOv2 전체 화면 특징 |
| T | 같은 16프레임 DINO 특징의 평균, 마지막-처음 차이, 평균 절대 프레임 차이 |
| C | 마지막 시점에 관측 가능한 기존 10프레임 객체 위치·크기·이동량 |
| V | V-JEPA 2 영상 특징 |
| GV | 정상 데이터로 각각 스케일을 맞춘 G와 V 점수의 최댓값 |

T는 저장된 특징으로 계산하며 추가 DINO 추론은 하지 않는다. C는 V와 시간 창 길이가 다르므로 참고 비교다. V는 외관·시간 정보가 함께 포함돼 순수한 동작 표현이라고 부르지 않는다. 영상 모델과 이미지 모델의 크기 및 해상도도 다르므로 시간 정보만의 효과를 완전히 분리한 실험은 아니다.

PCA residual과 가장 가까운 정상 사례의 거리 두 방식을 평가한다. 같은 학습 영상 01–27, 정상 scale 보정 28–30, 정상 임계값 설정 31–34를 유지한다. 학습·보정 특징도 같은 sparse endpoint에서 구성한다. 테스트 라벨을 사용해 fitting 또는 임계값을 설정하지 않는다.

주 비교는 모든 방법이 관측 가능한 동일 endpoint에서 한다. C 미관측은 0점으로 대체하지 않는다. sampled endpoint 비율과 공통 평가 비율, 영상별 평가 수를 함께 출력한다. **이 결과의 AUROC를 이전 전체 프레임 결과 0.855와 직접 비교하지 않는다.** 이번 실행에서 다시 계산한 G와 비교한다.

정상 구간에서 순서를 뒤집거나 마지막 프레임만 반복한 특징의 변화를 측정하는 probe도 저장한다. 이것은 정상 영상의 민감도 점검이며 이상 탐지 성능이나 시간 정보 활용의 충분한 증거는 아니다. 학습·점수 선택에 사용하지 않는다.

## 실행

기존 Colab의 새 셀에서 Drive를 연결하고 이 스크립트를 파일로 저장한 뒤 **새 Python 프로세스**로 실행한다. 이전 코드처럼 `exec`로 같은 Python 메모리에 섞지 않는다. V-JEPA 라이브러리는 `/content`의 별도 폴더에 고정 버전으로 설치하고 기존 torch 및 노트북의 CLIP/DINO 라이브러리는 교체하지 않는다.

```python
from google.colab import drive
from urllib.request import urlopen
from pathlib import Path
import subprocess, sys

drive.mount('/content/drive')
url = 'https://raw.githubusercontent.com/ljw-0312/ipad-ad-experiments/main/experiments/IPAD_VJEPA_CPU.py'
path = Path('/content/IPAD_VJEPA_CPU.py')
path.write_bytes(urlopen(url).read())
process = subprocess.Popen(
    [sys.executable, '-u', str(path)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f'V-JEPA process exited with code {code}; inspect the preceding log.')
except KeyboardInterrupt:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise
```

CPU 런타임을 선택한다. 최초 실행 시 가중치 다운로드 및 첫 clip 추론이 오래 걸릴 수 있다. 계산 시간은 측정 전 확정하지 않는다.

코랩에서는 `subprocess.run`의 기본 자식 프로세스 출력이 화면에 안 보일 수 있으므로, 위 코드는 표준 출력을 읽어 셀에 실시간으로 표시한다. 이전 셀이 계산 중이면 로그를 보기 위해 새 실행을 시작하지 않는다. Drive의 결과 보고서와 갱신되는 특징 캐시로 실행 상태를 확인할 수 있다.

각 clip이 완료될 때 Drive의 `meeting_experiments/vjepa_video_cache`에 저장한다. 중단 후 동일 셀을 실행하면 완료 clip을 다시 추론하지 않는다. ZIP의 R01 이미지가 임시 공간에서 없어지면 다시 복구하지만 GroundingDINO와 DINOv2를 다시 실행하지 않는다.

## 필요한 원본 자료

- `IPAD_project/IPAD_dataset.zip`
- `meeting_experiments/feature_cache`
- `R01_20261007_102640_886381_UTC/settings.json`
- `meeting_experiments/dinov2_feature_cache`
- `R01_20261007_102640_886381_UTC/dinov2_comparison_20261008_002809_792348_UTC/dinov2_report.json`

## 공유할 결과

원래 R01 결과 폴더 아래의 `vjepa_cpu_<timestamp>`에 결과를 생성한다.

- `vjepa_report.json`
- `temporal_comparison.png`

평가 결과는 모델 버전, 입력 설정, 샘플링, 관측 비율, 정상 데이터 분할과 함께 해석한다. sparse endpoint 실험은 전체 프레임 실험이나 실시간 성능 검증이 아니다.

참고: [Meta V-JEPA 2 코드](https://github.com/facebookresearch/vjepa2), [공식 HF 모델](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), [Transformers V-JEPA 2](https://huggingface.co/docs/transformers/model_doc/vjepa2).
