# MLE01. Python 엔지니어링 for ML

## 1. 객체 모델: 모든 버그의 출발점

### 1.1 변수는 상자가 아니라 이름표
Python에서 변수는 값을 담는 상자가 아니라 객체에 붙인 이름표임

```python
a = [1, 2, 3]       # 새 리스트가 아니라 같은 객체에 이름표를 하나 더 붙임
b = a
b.append(4)
print(a)            # [1, 2, 3, 4]
print(a is b)       # True: 같은 객체
```

- `is` 는 **같은 객체인가**(identity), `==` 는 **값이 같은가**(equality)를 비교함.
- 가변(mutable): `list`, `dict`, `set`, 대부분의 사용자 정의 객체, `torch.Tensor`, `np.ndarray`
- 불변(immutable): `int`, `float`, `str`, `tuple`, `frozenset`


### 1.2 얕은 복사 vs 깊은 복사

```python
import copy
config = {"lr": 1e-3, "layers": [64, 128]}

shallow = copy.copy(config)     # 최상위 객체만 새로 생성 (빠름)
deep = copy.deepcopy(config)    # 내포된 하위 객체까지 모두 새로 생성 (느림)

shallow["layers"].append(256)
print(config["layers"])     # [64, 128, 256] ← 얕은 복사는 내부 리스트를 공유
print(deep["layers"])       # [64, 128]
```

> ML 연결 : 하이퍼파라미터 탐색에서 base config를 얕은 복사한 뒤 내부 리스트를 수정하면,
> 이전 실험들의 config까지 오염됨. 실험 기록이 거짓말을 하게 되는 대표적인 버그임.


### 1.3 가변 기본 인자 함정

```python
# 나쁜 예
def add_metric(value, history=[]):
    history.append(value)
    return history

add_metric(0.9)     # [0.9]
add_metric(0.8)     # [0.9, 0.8] ← 기본 리스트가 호출 간에 공유됨!

# 좋은 예
def add_metric(value, history=None):
    if history is None:
        history = []
    history.append(value)
    return history
## python에서 자주 사용하는 관용적 표현 방법
```

> 기본 인자는 함수 정의 시점에 한 번만 평가되기 때문


## 1.4 텐서에서 같은 문제

```python
import torch
x = torch.zeros(3)
y = x.view(3)           # 메모리를 공유하는 뷰
y[0] = 1.0
print(x)                # tensor([1., 0., 0.])

z = x.clone()           # 독립된 복사본
```

> PyTorch의 `view`, 슬라이싱, NumPy의 슬라이싱의 대부분 메모리를 공유함.
> 이 감각은 MLE02, MLE12에서 계속 사용됨

---

## 2. 타입힌트: 코드에 계약서를 붙이기

### 2.1 기본

```python
from collections.abc import Sequence

def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty sequence")
    return sum(xs) / len(xs)

def find_ckpt(name: str) -> str | None :    # Python 3.10+ 문법
    ...
```

> 타입힌트는 런타임에 강제되지 않음.
> `mypy`나 `pyright` 같은 정적 검사기, 그리고 IDE 자동완성이 이를 활용함.

### 2.2 실무에 자주 쓰는 타입 도구

```python
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, TypeVar

# Literal: 허용 값 제한
Optimizer = Literal["adam", "sgd", "adamw"]

# TypeDict: 딕셔너리 형태 명시 (JSON 응답 등)
class Prediction(TypedDict):
    label: str
    score: float

# Protocol: "이런 메서드만 있으면 된다" (구조적 타이핑, duck typing의 정적 버전)
class Predictor(Protocol):
    def predict(self, x: list[float]) -> float: ...

def evaluate(model: Predictor, data: list[list[float]]) -> list[float]:
    return [model.predict(x) for x in data]

# TypeVar: 제너릭
T = TypeVar("T")
def first(items: list[T]) -> T:
    return items[0]
```

| 타입 | 내용 | 쉽게 설명 |
| ---- | ---- | -------- |
| dataclass | 데이터를 담는 객체를 만들기 쉽게 해주는 클래스 | 데이터 담는 클래스 |
| Literal | 특정 값들 중 하나만 허용 | A/B 중 하나만 |
| Protocol | 이런 메서드/속성을 가지고 있다면 이 타입으로 취급함 (상속이 아닌, 구조) | 이 메서드가 있으면 OK |
| TypedDict | 실제 새로운 Dictionary 클래스를 만드는 것이 아님 | 이 key/value를 가져야 함 |
| TypeVar | 제네릭(Generic)을 만들 때 사용하는 타입 변수 | 어떤 타입 T | 

> `Protocol`의 의미 : `evaluate`는 sklearn 모델이든, 직접 만든 클래스든, ONNX 래퍼든 `predict` 메서드만 있으면 받음
> 상속을 강요하지 않으므로 서로 다른 팀의 코드를 느슨하게 연결 가능.

### 2.3 Python 3.12+ 제네릭 문법

```python
def first[T](items: list[T]) -> T:
    return itmes[0]
```

> 제네릭(Generic) : 클래스나 함수 정의할 때 사용할 데이터 타입을 미리 고정하지 않고,
> 나중에 인스턴스를 생성하거나 사용할 때 구체적인 타입을 지정하는 프로그래밍 방식 

1. `TypeVar` : 제네릭에서 사용할 타입 변수를 정의 (`T = TypeVar('T')`)
2. `Generic` : 클래스를 제네릭으로 만들기 위해 상속받는 기본 클래스 (`classBox(Generic[T]):`)

---

## 3. 이터레이터와 제러레이터: 메모리보다 큰 데이터 다루기

### 3.1 이터레이터 프로토콜
`for x in obj`는 내부족으로 다음과 같이 동작함.

```python
it = iter(obj)          # obj.__iter__() 호출
while True:
    try:
        x = next(it)    # it.__next__() 호출
    except StopIteration:
        break
```

### 3.2 제너레이터: 필요할 때 하나씩 만든다

```python
def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# 100GB 파일도 한 줄씩만 메모리에 올라감
for record in read_jsonl("huge.jsonl"):
    process(record)
```

> `yield`를 만나면 함수는 값을 내보내고 그 자리에서 멈춘 상태를 기억함.
> 다음 `next()` 때 이어서 실행함.

### 3.3 제너레이터 파이프라인

```python
def read_lines(path):
    with open(paht, encoding="utf-8") as f:
        yield from f

def parse(lines):
    for line in lines:
        yield json.loads(line)

def filter_valid(records):
    for r in records:
        if r.get("text"):
            yield r

def batch(items, size):
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

for b in batch(filter_valid(parse(read_lines("data.jsonl"))), size=32):
    train_step(b)
```

> 각 단계가 지연평가(lazy)되므로 전체 메모리 사용량은 배치 하나 수준에 머뭄.
> 이 구조는 PyTorch의 `IterableDataset`, Hugging Face `datasets`의 streaming 모드와 같은 아이디어임.

### 3.4 리스트 컴프리헨션 vs. 제너레이터 표현식

```python
sum([x * x for x in range(10**8)])      # 리스트 전체를 메모리에 만든 후 합산
sum(x * x for x in range(10**8))        # 하나씩 생성하며 합산, 메모리 거의 0
```

> 주의: 제너레이터는 한 번만 소비됨. 두 번 순회하면 두 번째는 빈 사태임.
> 학습 루프에서 에폭(epoch)마다 다시 만들어야 함.

---

## 4. 클로저와 데코레이터

### 4.1 함수는 일급 객체
함수를 변수에 담고, 인자로 넘기고, 반환할 수 있음.
내부 함수가 바깥 함수의 변수를 기억하는 것은 "클로저(Closure)"라 함.

```python

```

```python

```