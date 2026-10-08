# DINOv2 / CLIP 외관 특징 비교

상태: 코드 준비 및 로컬 무결성 검사 완료. 실제 Colab DINOv2 추론과 전체 성능 결과는 실행 후 확인한다.

기존 Colab을 그대로 사용한다. CPU 또는 NVIDIA GPU에서 실행할 수 있다. TPU는 이 스크립트에서 지원하지 않는다. 원래 결과, feature_cache 및 IPAD_dataset.zip이 Drive에 있어야 한다. 런타임이 초기화되어도 원본 ZIP에서 R01만 복구한다.

## 목적

동일한 R01 프레임·객체 사각형·추적·정상 데이터 분할·점수 방법에서 CLIP과 frozen DINOv2-Small의 외관 특징을 비교한다. GroundingDINO는 재실행하지 않는다.

## 실험 구성

- 기존 CLIP 전체 화면 / 객체 crop 특징을 재사용한다.
- DINOv2-Small CLS 384차원 특징을 전체 화면과 동일 객체 crop에서 새로 추출하고 L2 정규화한다.
- 객체 crop 주변 여백은 5%로 동일하다.
- 이미지 전처리의 resize/center-crop 기하를 기존 CLIP과 맞추기 위해 shortest side 224, center crop 224를 사용한다. DINOv2 기본 processor의 shortest side 256을 224로 변경하며, 색상 정규화는 DINOv2 설정을 사용한다.
- PCA rank 16 및 정상 사례 거리 방식으로 각각 평가한다. 정상 점수 보정 그룹과 임계값 그룹도 동일하다.
- 동작 C는 기존 58차원 사각형 기록을 그대로 사용한다. 변경되지 않았는지 점수 수준에서 검사한다.
- A/B가 인코더 비교의 주요 결과다. D는 바뀐 외관과 기존 동작의 탐색적 결합 결과다.
- 검출 관측 비율과 공통 평가 프레임은 CLIP/DINOv2에서 정확히 같아야 한다.

## 실행 방법

`experiments/IPAD_DINOv2_Comparison.py`를 다운로드하고 같은 Colab의 새 셀에서 실행한다.

```python
from google.colab import drive, files
import os
# CPU로 실행할 때 지정한다. 자동 GPU/CPU 선택은 이 줄을 생략한다.
os.environ['IPAD_DEVICE'] = 'cpu'
drive.mount('/content/drive')
uploaded = files.upload()
exec(next(iter(uploaded.values())).decode('utf-8'))
```

또는 저장소의 Raw URL을 urllib로 읽어 실행할 수 있다. 저장소에 올린 파일의 commit SHA로 URL을 고정하면 재현하기 쉽다.

새 노트북을 만들거나 이전 결과를 삭제할 필요가 없다. DINOv2 특징은 별도 `dinov2_feature_cache`에 영상별로 저장한다. 중단 후 같은 설정으로 재실행하면 완료된 DINOv2 특징을 재사용한다.

CPU에서는 모델을 미세 조정하지 않고 float32 추론만 수행한다. 작은 이미지 batch와 최대 4개의 PyTorch CPU thread를 사용한다. 실제 전체 시간은 첫 영상 처리 속도로 추정한다. CPU 시간이 측정되지 않았으므로 몇 분 안에 끝난다고 보장하지 않는다. CPU/GPU는 동일 추출 설정의 캐시를 재사용하며 수치상 작은 차이가 있을 수 있다.

## 출력

원래 R01 결과 폴더 아래에 `dinov2_comparison_<UTC timestamp>`를 만든다.

- `dinov2_report.json`: 모델 revision, 환경, 처리 비용, 네 조건별 전체·영상별 평가.
- `encoder_comparison.png`: PCA / 정상 사례 거리에서 CLIP과 DINOv2의 비교.
- `clip_pca_residual`, `dinov2_pca_residual`, `clip_nearest_normal`, `dinov2_nearest_normal`: 각 조건의 metrics, frame scores 및 그래프.
- `comparison.csv`: 결과 표.

완료 후 `dinov2_report.json`과 `encoder_comparison.png`를 공유한다. 기존 결과와의 비교는 동일 공통 프레임 AUROC/AP 및 전체 이상 탐지율·정상 오경보율을 함께 본다.

## 검증 범위

로컬에서는 특징과 crop 순서, 사각형·추적·동작 기록 보존, 파일 해시, 캐시 재사용, 미관측 처리, 점수 비교 조건을 검사했다. 실제 pretrained DINOv2의 GPU 추론은 Colab 실행 초반에 검사한다. 아직 DINOv2 성능 개선이나 T4 처리 시간을 확인하지 않았다.

출처: [DINOv2 공식 코드](https://github.com/facebookresearch/dinov2), [Meta DINOv2-Small 모델](https://huggingface.co/facebook/dinov2-small).
