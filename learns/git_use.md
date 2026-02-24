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
git clone --recurse-submodules https://github.com/JJKim-617/3D-AD-Senior-Project.git
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

### 1-5. 서브모듈 추가하기

기존 repo에 다른 git repo를 서브모듈로 추가하는 방법.
외부 repo를 직접 서브모듈로 쓰기보다, **먼저 포크한 뒤 포크한 repo를 서브모듈로 등록**하는 것이 원칙.
포크 없이 원본 repo를 바로 등록하면 수정 사항을 push할 수 없음.

#### Step 0. GitHub에서 포크 먼저
1. 원본 repo 페이지 접속 (예: `https://github.com/eliahuhorwitz/3D-ADS`)
2. 우측 상단 `Fork` 버튼 클릭
3. 내 계정(`JJKim-617`)으로 포크 생성 → `https://github.com/JJKim-617/3D-ADS`

> 포크한 URL을 서브모듈 등록에 사용.

#### Step 1. 서브모듈 등록
```bash
# 상위 프로젝트 디렉토리에서 실행
cd /home/ryukimlee/3D-AD-Senior-Project

# git submodule add <repo_url> <로컬_경로>
git submodule add https://github.com/JJKim-617/3D-ADS.git external/3D-ADS
```

실행 후 자동으로 생성/수정되는 것들:
- `external/3D-ADS/` 디렉토리 생성 및 repo clone
- `.gitmodules` 파일 생성 (또는 내용 추가)

#### Step 2. .gitmodules 확인
```bash
cat .gitmodules
# 아래와 같이 등록되어 있어야 함
# [submodule "external/3D-ADS"]
#     path = external/3D-ADS
#     url = https://github.com/JJKim-617/3D-ADS.git
```

#### Step 3. 서브모듈 커밋 해시 기록 (상위 repo에 반영)
```bash
git add .gitmodules external/3D-ADS
git commit -m "Add 3D-ADS as submodule"
git push origin main
```

> 서브모듈 내부에서도 `git config --local user.name/email` 설정 필요 (1-1 참고).

**주의사항:**
| 상황 | 처리 방법 |
|---|---|
| 이미 존재하는 디렉토리에 추가 시 | 디렉토리를 먼저 삭제하거나 비운 후 실행 |
| Private repo를 서브모듈로 추가 시 | SSH URL 사용 권장 (`git@github.com:...`) |
| 서브모듈 URL 변경 시 | `.gitmodules` 수정 후 `git submodule sync` 실행 |

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
