# tmux 사용법 (SSH 세션 유지)

SSH 연결이 끊겨도 프로세스가 계속 실행되도록 tmux 사용.

## 기본 흐름

### 1. tmux 세션 시작
```bash
tmux new -s <세션이름>
```

### 2. 작업 실행
```bash
source /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS/3d_ads_venv/bin/activate
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS
CUDA_VISIBLE_DEVICES=1 python main.py
```

### 3. Detach (SSH 끊어도 세션 유지)
```
Control+B 누르고 손 떼고 → D 누르기
```
터미널에 `[detached (from session <이름>)]` 뜨면 성공.

### 4. 재접속 후 세션 다시 붙기
```bash
tmux attach -t <세션이름>   # 특정 세션
tmux ls                     # 실행 중인 세션 목록 확인
```

---

## 자주 쓰는 단축키

> 맥 기준 Control (⌃), Windows/Linux 기준 Ctrl

| 키 | 동작 |
|----|------|
| `Control+B, D` | detach (세션 유지하며 나가기) |
| `Control+B, [` | 스크롤 모드 (로그 확인) |
| `q` | 스크롤 모드 종료 |
| `Control+B, &` | 세션 강제 종료 |

---

## 3D-ADS 실행 예시

```bash
tmux new -s ads
source /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS/3d_ads_venv/bin/activate
cd /home/ryukimlee/3D-AD-Senior-Project/external/3D-ADS
CUDA_VISIBLE_DEVICES=1 python main.py
# Control+B, D 로 detach
```

재접속:
```bash
tmux attach -t ads
```
