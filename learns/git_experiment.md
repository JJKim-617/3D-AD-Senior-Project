# 실험 관리 - Git Branch 전략

BTF 관련 실험은 서브모듈인 `external/3D-ADS`에서 브랜치를 분리하여 진행.

---

## 1. 실험 브랜치 구조

```
JJKim-617/3D-ADS (포크한 repo)
│
├── main                          ← 초기 설정 상태 (기준점)
├── experiment/fpfh-voxel-tuning  ← 실험 A
├── experiment/add-normal-feature ← 실험 B
└── experiment/preprocessing      ← 실험 C
```

`main`은 항상 안정적인 기준 상태를 유지.
실험은 각자 독립된 브랜치에서 진행 후, 좋은 결과가 나오면 `main`에 merge.

---

## 2. 실험 시작 방법

```bash
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS

# 항상 main에서 분기
git checkout main
git checkout -b experiment/실험이름

# 예시
git checkout -b experiment/fpfh-voxel-tuning
```

---

## 3. 실험 중 커밋/푸시

```bash
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS

git add .
git commit -m "experiment: 변경 내용 설명"
git push origin experiment/실험이름
```

---

## 4. 실험 완료 후 main에 merge

```bash
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS

git checkout main
git merge experiment/실험이름
git push origin main
```

---

## 5. 상위 프로젝트(Senior Project)에 반영

서브모듈 push 후, 상위 repo에도 커밋 해시 업데이트 필요.

```bash
cd /home/ryukimlee/3D-AD-Senior-Project

git add external/3D-ADS
git commit -m "update 3D-ADS submodule: 실험 내용 요약"
git push origin main
```

---

## 6. 브랜치 목록 확인

```bash
# 로컬 브랜치
git branch

# 원격 브랜치 포함
git branch -a
```

---

## 7. 실험 네이밍 컨벤션 예시

| 브랜치명 | 설명 |
|---|---|
| `experiment/fpfh-voxel-tuning` | FPFH voxel_size 파라미터 실험 |
| `experiment/add-normal-feature` | 법선벡터 feature 추가 실험 |
| `experiment/rgb-backbone-swap` | Backbone 모델 교체 실험 |
| `experiment/preprocessing-test` | 전처리 적용 효과 실험 |
