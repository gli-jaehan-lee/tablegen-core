# 훈련용 데이터 생성기 사용법

**GUI 없이 Python에서 표를 만들고, 배열을 모델에 전달하면 됩니다.** 생성기는 모델을 훈련시키지 않습니다. 아래 예제는 사용자가 작성할 훈련 루프를 보여주는 작은 CPU 데모입니다.

## 1. 설치와 첫 훈련

소스 저장소를 내려받아 그 폴더에서 실행합니다. Python 3.10 이상이 필요합니다. Tk는 GUI에만 필요합니다.

```bash
python3 -m venv ~/.venvs/tablegen_core-training
source ~/.venvs/tablegen_core-training/bin/activate
python -m pip install .
python examples/training_quickstart.py
```

Windows에서는 가상환경 활성화 명령을 `경로\Scripts\activate`로 바꾸세요. 최초 의존성 설치에는 인터넷과 시간이 필요합니다. 예제 자체는 CPU 한 스레드, 기본 128행·4입력·1정답·100회 업데이트입니다.

```bash
python examples/training_quickstart.py --rows 512 --features 8 --seed 73 --steps 200 --output output/train-demo
```

출력의 `parameters_changed: true`는 실제 역전파/가중치 갱신을 뜻합니다. `train_mse_standardized`, `eval_mse_standardized`는 **학습 행의 정답 표준편차 단위**로 측정한 평균제곱오차입니다. 좋은 연구 성능을 인증하는 점수가 아닙니다. `--output`을 지정하면 새 폴더에 원본 표와 메타데이터, `training_metrics.json`을 남깁니다. 기존 폴더는 덮어쓰지 않습니다.

## 2. 내 훈련 코드에 붙이기

```python
from tablegen_core import generate

result = generate(mode="direct", priors=["lin"], rows=1000,
                  features=8, targets=1, seed=42)
X = result.data[:, :8]   # (1000, 8), 입력 열
Y = result.data[:, 8:]   # (1000, 1), 정답 열; 다중 정답도 2차원 유지
# 파일이 필요할 때만:
# result.write("output/new-table")
```

`mode="direct"`는 프라이어 하나를 직접 사용합니다. `mode="dag"`는 그래프를 따라 여러 생성 규칙을 연결합니다. 혼합은 DAG에서 `priors=["default"]` 등으로 지정합니다. 숫자 코드로 표현한 범주 열을 연속형 수치처럼 학습시키지 않도록 모델에 맞는 임베딩/인코딩을 선택하세요. 위 실행 예제는 수치 회귀만 다룹니다.

## 3. 훈련/평가 경계와 재현성

- 예제는 **한 번 생성한 표**의 앞 75%를 학습, 뒤 25%를 평가에 사용합니다. 입력과 정답의 평균·표준편차는 학습 부분에서만 계산하고 평가 부분에는 같은 변환을 적용합니다. 평가 정답은 역전파에 들어가지 않습니다.
- 이것은 같은 생성 문제 내의 보류 행 평가입니다. 새로운 표/과제에 대한 일반화 평가와 다릅니다. 모든 프라이어가 독립·동일 분포 행을 만든다는 보장은 없습니다. 시간·그룹 구조가 있으면 그 구조에 맞는 분할을 직접 설계하세요. 단순 행 섞기는 독립성을 보장하지 않습니다.
- 새로운 과제마다 새로운 시드를 사용하세요. 서로 다른 훈련/평가 과제 목록에는 겹치지 않는 시드 범위를 배정하세요. 예: 훈련 0–999, 평가 10000–10099. 같은 시드와 설정으로 다시 생성해 두 개의 독립 표라고 부르면 안 됩니다. 시드 분리만으로 의미적으로 다른 분포가 보장되지는 않습니다.
- 패키지 버전, 전체 생성 설정, NumPy/SciPy/PyTorch 버전, 시드 목록을 함께 보관하세요. 다른 버전/장치까지 비트 단위 동일함을 약속하지 않습니다. CSV보다 정확한 재사용에는 `table.npy`와 `metadata.json`을 함께 보관하세요.
- 이 빠른 예제는 체크포인트 복구를 구현하지 않습니다. 실제 훈련에는 모델·옵티마이저·난수 상태·다음 데이터 번호를 함께 저장하세요. 스레드 병렬화보다 각 프로세스가 서로 겹치지 않는 데이터 번호를 소유하도록 설계하세요.

실행 코드: [training_quickstart.py](examples/training_quickstart.py). 프로젝트 안내: [README](README.md).
