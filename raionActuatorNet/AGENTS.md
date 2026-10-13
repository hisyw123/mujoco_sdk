# 작업 규칙

- 이 디렉터리는 RaiSim 원본의 CSV 처리 및 GRU 학습 방식을 최대한 유지하는 MuJoCo 이식본이다.
- 현재 기본 물리/수집/추론/뷰어는 build/augmented_sim(C++, 기존 MuJoCo 3.13.1 링크)이다.
  Python MuJoCo 3.8.1 코드는 레거시 비교용이며 기본 실행부로 되돌리지 않는다. 최신 실행법은 NATIVE_CPP.md.
- RoughTerrain 선택은 원본을 보존하는 stand27 복사본이다. 자유베이스/그리퍼 고정/접촉 비활성화 전제를 명시한다.
- reference/raisim은 원본 보존본이다. 수정하지 않는다.
- 작업마다 `patch note.txt`에 날짜, 구현 위치, 기능, 원본 대비 변경, 검증 결과, 미완료 사항을 추가한다.
- 변경 후 `.venv/bin/python scripts/audit_changes.py`로 timestamp별 diff/SHA256을 누적한다.
- 엔진 PD/적분 차이를 숨기지 않는다. 합성 로그와 실기 로그, 연결 테스트와 성능 검증을 구분한다.
- 원본 학습/분할 방식을 변경할 때에는 명시하고 기록한다. 기존 출력/사용자 로봇 모델을 덮어쓰지 않는다.
