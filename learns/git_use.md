# Git 사용 방법 및 구조

---

## 1. Git 사용 방법

### 1-1. 사용자 등록 (로컬 설정)

이 프로젝트는 공유 리눅스 계정(`ryukimlee`)을 사용하므로 반드시 `--local` 옵션으로 설정.
`--global`은 같은 계정을 쓰는 다른 사람에게도 영향을 주므로 사용 금지.

```bash
# 각 repo 디렉토리 안에서 실행
git config --local user.name "유저명"
git config --local user.email "GitHub에_등록된_이메일"

# 설정 확인
git config --local --list
```

> 서브모듈(`external/3D-ADS`)과 상위 프로젝트(`3D-AD-Senior-Project`) 각각 설정 필요.

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

### 1-3. 서브모듈 포함 시 add/commit/push 순서

서브모듈과 상위 repo는 **독립적으로** 관리. 반드시 서브모듈 먼저 처리.

#### Step 1. 서브모듈 변경사항 처리 (3D-ADS)
```bash
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS

git add .
git commit -m "커밋 메시지"
git push origin main          # 포크(JJKim-617/3D-ADS)에 push
```

#### Step 2. 상위 프로젝트 처리 (Senior Project)
```bash
cd /home/ryukimlee/3D-AD-Senior-Project

git add external/3D-ADS       # 서브모듈의 새 커밋 해시 기록
git add .                     # 그 외 변경된 파일
git commit -m "커밋 메시지"
git push origin main          # JJKim-617/3D-AD-Senior-Project에 push
```

> 상위 repo는 서브모듈의 **코드 내용이 아닌, 어떤 커밋을 가리키는지(커밋 해시)** 만 기록.

---

### 1-4. .gitignore에 들어가야 하는 것들

#### 상위 프로젝트의 .gitignore (`3D-AD-Senior-Project/.gitignore`)
```
# 데이터셋 (용량이 크고 공유 불필요)
Datasets/
```

#### 서브모듈의 .gitignore (`external/3D-ADS/.gitignore`)
```
# 가상환경 (OS/환경마다 달라서 공유 의미 없음, requirements.txt로 재생성 가능)
3d_ads_venv/

# pyenv 로컬 python 버전 파일 (이미 .gitignore에 포함됨)
.python-version
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
│
└── [submodule] JJKim-617/3D-ADS         ← 포크한 3D-ADS repo
    (원본: eliahuhorwitz/3D-ADS)
```

로컬 디렉토리:
```
3D-AD-Senior-Project/          ← git repo (JJKim-617/3D-AD-Senior-Project)
├── .git/
├── .gitmodules                ← 서브모듈 등록 정보
├── .gitignore
├── Datasets/                  ← gitignore 처리 (데이터셋)
├── learns/                    ← 학습 문서
└── external/
    └── 3D-ADS/                ← git submodule (JJKim-617/3D-ADS)
        ├── .git/
        ├── .gitignore
        ├── 3d_ads_venv/       ← gitignore 처리 (가상환경)
        ├── data/
        ├── feature_extractors/
        ├── utils/
        ├── main.py
        ├── patchcore_runner.py
        └── requirements.txt
```

---

### 2-2. 서브모듈 (git submodule)

`.gitmodules` 내용:
```
[submodule "external/3D-ADS"]
    path = external/3D-ADS
    url = https://github.com/JJKim-617/3D-ADS.git
```

**서브모듈이란?**
- 상위 repo 안에 다른 git repo를 포함시키는 방식
- 상위 repo는 서브모듈의 특정 커밋 해시만 추적
- 서브모듈 내부는 완전히 독립된 git repo로 동작

**서브모듈 처음 clone 시 (다른 환경에서 받을 때):**
```bash
git clone --recurse-submodules https://github.com/JJKim-617/3D-AD-Senior-Project.git

# 또는 clone 후
git submodule update --init --recursive
```

---

### 2-3. 포크 (Fork)

- **원본**: `eliahuhorwitz/3D-ADS` (논문 저자)
- **포크**: `JJKim-617/3D-ADS` (내 계정)

**포크를 사용하는 이유:**
- 원본 코드를 기반으로 자유롭게 수정 가능
- 수정 단계별로 내 repo에 커밋/푸시 가능
- 원본 저자의 업데이트와 독립적으로 실험 관리

**원본 저자의 커밋 히스토리 반영 방지:**
포크 후에는 원본과 독립적이므로 원본의 새 커밋이 자동 반영되지 않음.
필요 시 수동으로 upstream을 등록하여 동기화 가능:
```bash
git remote add upstream https://github.com/eliahuhorwitz/3D-ADS.git
git fetch upstream
git merge upstream/main
```
