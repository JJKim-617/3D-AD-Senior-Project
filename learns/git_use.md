# Git 사용 방법 및 구조

---

## 0. 협업 워크플로우 (가장 중요)

### 왜 같은 서버 경로를 공유하면 안 되는가

같은 리눅스 계정 + 같은 디렉토리를 두 사람이 공유하면:

- **파일시스템을 공유**: 한 사람이 만든 파일/폴더는 `git add/commit` 전이라도 상대방에게 즉시 보임
- **브랜치 상태를 공유**: `.git/HEAD` 파일이 하나이므로, 한 사람이 `git switch 브랜치` 하면 상대방 작업 환경도 바뀜
- **작업 파일 충돌**: `git restore`, `git checkout` 등 명령어가 상대방 작업 중인 파일을 덮어쓸 수 있음

브랜치를 각자 따로 만들어도, 같은 경로를 쓰는 한 위 문제는 해결되지 않는다.

### 권장 워크플로우

```
[로컬 PC]                          [SSH 서버 (GPU)]
코드 작성 (VSCode 등)               실험 실행
git push origin 브랜치   →  git pull origin 브랜치  →  python main.py
```

- **로컬**: 코드 작성, 디버깅
- **SSH 서버**: 학습/실험 실행 (GPU 활용)
- **각자 계정에 별도 clone**: 서로 영향 없이 독립적으로 작업

```bash
# 각자 본인 계정 home에 clone
git clone https://github.com/JJKim-617/3D-AD-Senior-Project.git
```

### 실제 흐름

```bash
# [로컬] 코드 작성 후
git add .
git commit -m "feat: ..."
git push origin my-branch

# [SSH 서버] 실험 실행
git pull origin my-branch
CUDA_VISIBLE_DEVICES=1 python main.py 2>&1 | tee run_log.txt
```

---

## 1. Git 사용 방법

### 1-1. 사용자 등록 (로컬 설정)

이 프로젝트는 공유 리눅스 계정(`ryukimlee`)을 사용하므로 반드시 `--local` 옵션으로 설정.
`--global`은 같은 계정을 쓰는 다른 사람에게도 영향을 주므로 사용 금지.

```bash
# repo 디렉토리 안에서 실행
git config --local user.name "유저명"
git config --local user.email "GitHub에_등록된_이메일"

# 설정 확인
git config --local --list
```

---

### 1-2. 브랜치 만들기

```bash
# 새 브랜치 생성 및 이동
git checkout -b 브랜치이름

# 예시: 실험 단위로 브랜치 분리
git checkout -b experiment/fpfh-tuning

# 현재 브랜치 확인
git branch

# 브랜치 전환
git switch 전환하고자_하는_브랜치

# 원격에 브랜치 push
git push origin 브랜치이름
```

---

### 1-3. add/commit/push 순서

```bash
cd /home/ryukimlee/3D-AD-Senior-Project

git add .                     # 변경된 파일 스테이징
git commit -m "커밋 메시지"
git push origin main          # JJKim-617/3D-AD-Senior-Project에 push
```

---

### 1-4. .gitignore에 들어가야 하는 것들

#### 상위 프로젝트의 .gitignore (`3D-AD-Senior-Project/.gitignore`)
```
# 데이터셋 (용량이 크고 공유 불필요)
Datasets/

# 가상환경 (OS/환경마다 달라서 공유 의미 없음, requirements.txt로 재생성 가능)
external/3D-ADS/3d_ads_venv/
```

**gitignore 원칙:**
| 항목 | 이유 |
|---|---|
| 가상환경 폴더 (`venv/`, `3d_ads_venv/` 등) | 용량 크고 환경마다 달라 공유 불필요 |
| 데이터셋 (`Datasets/`) | 수십~수백 GB, git으로 관리 부적합 |
| 캐시/빌드 결과물 (`__pycache__/`, `*.pyc`) | 자동 생성됨 |
| 모델 weight 파일 (`*.pth`, `*.pt`) | 용량 크고 코드로 재현 가능 |

---

## 2. Git 구조

### 2-1. 전체 구조

```
GitHub: JJKim-617/3D-AD-Senior-Project   ← 메인 프로젝트 repo
```

로컬 디렉토리:
```
3D-AD-Senior-Project/          ← git repo (JJKim-617/3D-AD-Senior-Project)
├── .git/
├── .gitignore
├── Datasets/                  ← gitignore 처리 (데이터셋)
├── learns/                    ← 학습 문서
└── external/
    └── 3D-ADS/                ← 일반 파일 (원본: eliahuhorwitz/3D-ADS)
        ├── 3d_ads_venv/       ← gitignore 처리 (가상환경)
        ├── data/
        ├── feature_extractors/
        ├── utils/
        ├── main.py
        ├── patchcore_runner.py
        └── requirements.txt
```

> `external/3D-ADS`는 별도 git repo가 아닌 일반 파일로 관리됨.
> 원본 코드를 수정하면 상위 프로젝트에서 바로 `git add/commit`으로 기록.

---

### 2-2. 포크 (Fork) 원본 참고

- **원본**: `eliahuhorwitz/3D-ADS` (논문 저자)
- **참고용**: `JJKim-617/3D-ADS` (초기 포크, 현재는 상위 프로젝트에 통합)

**원본 저자의 최신 변경사항이 필요한 경우:**
```bash
cd external/3D-ADS
git init
git remote add upstream https://github.com/eliahuhorwitz/3D-ADS.git
git fetch upstream
git checkout upstream/main -- .   # 원본 파일로 덮어쓰기
```
