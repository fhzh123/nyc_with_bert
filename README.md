# nyc_with_bert

nyc_with_bert

Dataset : https://drive.google.com/drive/folders/14nKtjfIjFNC_iZFw-XwVvHMFQ0uDXl6e?usp=sharing

## 해야할 일들 정리

1. **Taxi 수요 예측 관련 논문 리스트화 및 요약 정리 (기왕이면 엑셀이나 마크다운에)**

* reference 에 정리중, STDN, TGNET 먼저 정리예정

* 목표:

> Q1. 특정 요일 / 시간대가 주어질 때, 어느 지역에서 택시의 수요가 많을 것인가?

>  Q2. 특정 요일 / 시간대 / 위치 정보가 주어질 때 해당 지역의 택시 수요는 어느정도 일 것인가?

>  Q3. 택시 수요를 예측하는데 걸리는 시간(속도)는 어느정도 인가? --> Business goal : 택시에게 앞으로 수요증가가 예측되는 지점을 알려줄 수 있을것인가? 또는 택시를 배분할 수 있을 것인가?




2. **Input Output 시각화 (논문에도 쓰고 날 이해도 시킬 용도)**

* Input : a*b Grid, 30분 단위로 일주일(또는 하루)데이터 --> map shapefile 은 있으나, traffic 데이터를 어떻게 활용해야함? 

>  D1. 공간 & 트래픽 정보 : 뉴욕지도(grid 나눈값 못찾으면 구역으로 나눈 ID값 우선 적용), 트래픽정보 (30분단위)

> D2. 시간 embedding : 데이터 형태 첨부

> D3. 해당지역 트래픽 예측 embedding : output에 적용




3. **모델 세부내용을 제외한 기초 코드 (이거 주말동안 내가 짜놓을게)**

   


4. **상세 일정 확정 (이게 내 경험상으론 생각보다 중요한게 상세히 일정 안잡으면 뒤로 한없이 쳐짐)**

