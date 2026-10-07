## 1. ML 엔지니어라는 직무 지도

### 1.1 유사한 직군들

| 직무 | 핵심 질문 | 주요 산출물 | 주로 쓰는 역량 |
| --- | --------- | ---------- | ------------- |
| Research Scientist | "새로운 방법이 기존보다 나은가?" | 논문, 새 알고리즘 | 수학, 실험 설계 |
| ML Engineer | "이 모델을 실제 환경에서 안정적으로, 빠르게, 싸게 돌릴 수 있는가?" | 학습 파이프라인, 서빙 시스템, 경량화된 모델 | 소프트웨어 공학 + ML |
| Data Scientist | "데이터가 비즈니스에 무엇을 말하는가?" | 분석, 실험 결과, 의사결정 근거 | 통계, 도메인 |
| MLOps Engineer | "모델 생애주기를 자동화할 수 있는가?" | CI/CD, 모니터링, 인프라 | DevOps 클라우드 | 

> Research Scientist가 엔진을 설계한다면,
> ML 엔지니어는 그 엔진을 실제 자동차에 얹어서 10만 km를 고장 없이 달리게 만드는 사람

---

### 1.2 기업 유형별 MLE의 실제 모습

| 기업명 | 기능 | 핵심 단어 | 
| 빅테크 (구글) | 수십억 사용자 트래픽, 분산 학습, 지연시간 SLA | "대규모" |
| 전자·반도체 (삼성전자, LG) | 스마트폰·가전의 NPU 위에서 돌아가는 온디바이스 모델, 반도체 공정 데이터 분석 | "경량화·하드웨어 제약" | 
| 방산·모빌리티 (한화에어로스페이스, 현대로템) | 센서데이터, 실시간 엣지 추론, 안전중요 시스템, 폐쇄망 환경 | "신뢰성·실시간성·오프라인 배포" | 

> 실무 포인트
> 방산 기업은 보안상 인터넷이 차단된 폐쇄망에서 개발하는 경우가 많음
> `pip install`이 안 되는 환경에서 의존성을 오프라인으로 옮기는 능력(wheel 다운로드, Docker 이미지 반입)이 실제로 중요

--- 

### 1.3 커리큘럼의 핵심 사이클

① 개념 읽기 (직관 → 수식 → 코드)
② 직접 구현 (라이브러리 없이 한 번, 라이브러리로 한 번)
③ 연습문제 (Level 1 → 4)
④ 회고 노트 (모르는 것, 틀린 것, 실무 연결점)

---

## 2. Linux & Shell: GPU 서버를 다루는 최소 무기

ML 학습은 대부분 원격 Linux 서버에서 발생함. 로컬 노트북이 아니라 서버를 편하게 다루는 게 첫 번째 실무 체력임.


### 2.1 반드시 손에 익힐 명령어
 
| 범주 | 명령어 | 용도 |
|---|---|---|
| 탐색 | `pwd`, `ls -alh`, `cd`, `tree -L 2` | 위치·파일 확인 |
| 파일 | `cp -r`, `mv`, `rm -r`, `mkdir -p`, `ln -s` | 복사·이동·심볼릭 링크 |
| 내용 | `cat`, `head -n`, `tail -f`, `less`, `wc -l` | 로그 확인, 데이터 줄 수 |
| 검색 | `grep -rn "pattern" .`, `find . -name "*.py"` | 코드·파일 검색 |
| 디스크 | `df -h`, `du -sh *` | 데이터셋이 디스크를 얼마나 먹는지 |
| 프로세스 | `ps aux`, `top`/`htop`, `kill -9 PID` | 멈춘 학습 종료 |
| GPU | `nvidia-smi`, `watch -n 1 nvidia-smi` | GPU 사용률·메모리 |
| 원격 | `ssh user@host`, `scp`, `rsync -avP` | 접속, 파일 전송 |
| 세션 | `tmux new -s train`, `tmux attach -t train` | 접속이 끊겨도 학습 유지 |


### 2.2 파이프와 리다이렉션

쉘의 진짜 힘은 작은 도구를 조합하는데 있음.

```bash
# 학습 로그에서 val_loss 줄만 뽑아 마지막 5개 보기
grep "val_loss" train.log | tail -n 5

# 표준출력과 에러를 모두 파일에 기록하면서 화면에도 출력
python train.py 2>&1 | tee train.log

# CSV와 두 번째 열에서 고유값 개수 세기 (헤더 제외)
tail -n +2 data.csv | cut d',' -f2 | sort | uniq -c | sort -rn | head
```

- `>`는 덮어쓰기, `>>`는 이어쓰기
- `2>&1`은 표준에러(2)를 표준출력(1)과 같은 곳으로 보냄
- `|`는 앞 명령의 출력을 뒤 명령의 입력으로 연결


### 2.3 권한

`ls -l` 결과의 `-rwxr-xr--` 는 소유자/그룹/기타 순서로 읽기(r=4)·쓰기(w=2)·실행(x=1) 권한입니다.
 
- `chmod 755 run.sh` → 소유자 rwx(7), 그룹 r-x(5), 기타 r-x(5)
- `chmod +x run.sh` → 실행 권한 추가


### 2.4 환경변수

```bash
export CUDA_VISIBLE_DEVICES=0,1   # 0, 1번 GPU만 프로세스에 보이게 함
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2 python train.py   # 이 명령에만 적용
```
 
`CUDA_VISIBLE_DEVICES=2` 로 실행하면 프로세스 안에서는 물리 2번 GPU가 `cuda:0` 으로 보입니다. 공유 서버에서 동료의 GPU를 침범하지 않는 기본 매너임.

 
### 2.5 접속이 끊겨도 학습 유지하기
 
SSH가 끊기면 그 세션에서 실행한 프로세스도 기본적으로 종료됨. 해결책은 두 가지
 
```bash
# 방법 1: tmux (권장)
tmux new -s exp01
python train.py          # 실행 후 Ctrl+b, d 로 빠져나옴(detach)
tmux ls                  # 세션 목록
tmux attach -t exp01     # 다시 붙기
 
# 방법 2: nohup
nohup python train.py > train.log 2>&1 &
```
 
---

## 3. Git : 혼자 쓸때와 팀에서 쓸 때

### 3.1 세 영역 모델

```
작업 디렉터리 --(git add)--> 스테이징 영역 --(git commit)--> 로컬 저장소 --(git push)--> 원격 저장소
```


### 3.2 실무 협업 흐름
 
```bash
git switch -c feat/add-augmentation     # 기능 브랜치 생성
# ... 작업 ...
git add src/data/augment.py
git commit -m "feat(data): add random erasing augmentation"
git fetch origin
git rebase origin/main                  # 최신 main 위로 내 커밋을 재배치
git push -u origin feat/add-augmentation
# → GitHub에서 Pull Request 생성, 코드 리뷰 후 머지
```

### 3.3 merge vs rebase
 
| | merge | rebase |
|---|---|---|
| 이력 | 머지 커밋이 생기고 분기 이력 보존 | 일직선 이력 |
| 위험 | 낮음 | **이미 push해서 남과 공유한 커밋은 rebase 금지** (이력이 바뀌어 동료 저장소와 충돌) |
| 사용처 | main에 PR 머지 | 내 개인 브랜치를 최신화할 때 |


### 3.4 ML 프로젝트의 Git 규칙
 
- **데이터와 모델 가중치는 Git에 넣지 않는다.** Git은 텍스트 diff용. 수 GB의 `.pt` 파일을 넣으면 저장소가 영구히 무거워짐(히스토리에 남기 때문).
- 대용량은 Git LFS, DVC(MLE29에서 상세), 또는 오브젝트 스토리지(S3/GCS)에 두고 **경로와 해시만** Git에 기록함.
- 비밀 정보(API 키)는 `.env` 에 두고 `.gitignore` 에 등록함.

```gitignore
# .gitignore 예시
__pycache__/
*.pyc
.venv/
.env
data/
outputs/
wandb/
*.pt
*.ckpt
.ipynb_checkpoints/
```
 
> **실무 포인트**: 실수로 API 키를 커밋하고 push했다면, 커밋을 지우는 것만으로는 부족함. 이미 노출된 것으로 간주하고 **키를 즉시 폐기·재발급**해야 함.


### 3.5 커밋 메시지 컨벤션 (Conventional Commits)
 
`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:` 접두어를 씀. 실험 이력을 나중에 추적할 때 큰 도움이 됨!!
 
| 접두어 | 의미 | 언제 사용 | 예시 |
| `feat:` | Feature | 새로운 기능 추가 | `feat: 회원가입 기능 추가` |
| `fix:` | Bug Fix | 버그 수정 | `fix: 로그인 오류 수정` |
| `refactor:` | Refactoring | 기능 변화 없이 코드 구조 개선 | `refactor: 사용자 인증 로직 분리` |
| `test:` | Test | 테스트 코드 추가/수정 | `test: 로그인 API 테스트 추가` |
| `docs:` | Documentation | README, API, 개발 문서 수정 | `docs: API 사용법 추가` |
| `chore:` | Chore | 기능과 직접 관계없는 잡일/설정 변경 (패키지, 빌드, CI/CD 설정, `.gitignore` 등) | `chore: 의존성 패키지 업데이트` |
| `perf:` | Performance | 성능 개선 | `perf: DB 조회 쿼리 최적화` |

---

## 4. Python 환경 관리: "제 컴퓨터에서는 안되는데요?"를 없애기

### 4.1 왜 가상환경이 필요한가?
프로젝트 A는 `torch==2.1`, 프로젝트 B는 `torch==2.5` 가 필요할 수 있습니다. 시스템 Python 하나에 모두 설치하면 충돌합니다. 가상환경은 **프로젝트마다 독립된 패키지 공간**을 만듬


### 4.2 도구 비교
 
| 도구 | 특징 | 언제 쓰나 |
|---|---|---|
| `venv` + `pip` | 표준 라이브러리, 가장 단순 | 가벼운 프로젝트 |
| `uv` | Rust로 작성된 매우 빠른 설치·잠금 도구, `pyproject.toml` 중심 | 최신 Python 프로젝트 기본값으로 추천 |
| `conda` / `mamba` | Python 외 바이너리(CUDA 라이브러리 등)까지 관리 | 복잡한 과학계산·시스템 의존성 |

```bash
# uv 기본 흐름
uv init mle-project
cd mle-project
uv add torch numpy pandas
uv add --dev pytest ruff
uv run python train.py
uv lock            # uv.lock 에 정확한 버전 고정
```


### 4.3 버전 명시와 잠금 파일
 
- `pyproject.toml`: "이 범위면 된다"는 **의도** (예: `numpy>=1.26`)
- 잠금 파일(`uv.lock`, `requirements.lock` 등): "실제로 이 버전이 깔렸다"는 **사실**
재현성을 위해 둘 다 Git에 커밋함.
 
---

## 5. Docker: 환경 전체를 상자에 담기

### 5.1 핵심 개념
- **이미지(Image)**: 실행 환경의 스냅샷(설계도). 읽기 전용 레이어의 묶음.
- **컨테이너(Container)**: 이미지를 실행한 인스턴스(실제 건물).
- **레이어 캐시**: Dockerfile의 각 명령이 레이어가 되며, 바뀌지 않은 레이어는 재사용됨.


### 5.2 ML용 Dockerfile 예시
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
 
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# ① 의존성 먼저 복사 → 코드만 바뀌면 이 레이어는 캐시 재사용
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
 
# ② 자주 바뀌는 코드는 마지막에
COPY src/ ./src/
 
CMD ["python3", "-m", "src.train"]
```

> **레이어 순서 원칙**: 자주 안 바뀌는 것(OS, 의존성)은 위에, 자주 바뀌는 것(코드)은 아래에 위치. 순서를 반대로 하면 코드 한 줄만 고쳐도 패키지를 전부 다시 설치함.


## 5.3 실행
 
```bash
docker build -t mle-train:0.1 .
docker run --gpus all -v $(pwd)/data:/app/data --shm-size=8g mle-train:0.1
```

- `--gpus all`: 호스트에 NVIDIA Container Toolkit이 설치되어 있어야 컨테이너가 GPU를 씀.
- `-v`: 데이터는 이미지에 굽지 않고 **볼륨으로 마운트**함.
- `--shm-size`: PyTorch DataLoader가 `num_workers>0` 일 때 공유메모리(`/dev/shm`)를 사용합니다. 기본값이 작아서 `bus error` 가 나는 경우가 흔함.


### 5.4 폐쇄망 반입
 
```bash
docker save mle-train:0.1 | gzip > mle-train.tar.gz   # 인터넷 되는 곳에서
docker load < mle-train.tar.gz                         # 폐쇄망 서버에서
```
 
---