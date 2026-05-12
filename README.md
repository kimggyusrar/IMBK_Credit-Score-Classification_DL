# 프로젝트명 : 신용점수 분류 DL 인사이트 분석 

---
## period : 2026.05.12 

## Tech Stack

### Language & Environment
* Language: Python
* Environment: Jupyter Notebook, Google Colab

### Data Science & Deep Learning
* Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
* Feature Engineering & Selection: Scikit-learn (mutual_info_classif)
* Deep Learning: PyTorch, pytorch-tabnet (TabNetClassifier, TabNetPretrainer)
* Machine Learning: LightGBM
* Ensemble: Soft Voting (TabNet + LightGBM 가중치 최적화)
* Preprocessing: LabelEncoder, QuantileTransformer, StandardScaler

## Data Source
* Kaggle Credit Score Classification Dataset (train2.csv)

출처: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
데이터 클리닝 참고: https://www.kaggle.com/code/clkmuhammed/credit-score-classification-part-1-data-cleaning


## Data Preprocessing
데이터 품질을 높이고 모델의 왜곡을 방지하기 위해 아래와 같이 단계별 전처리를 수행했습니다.

Feature Cleaning: 식별자 칼럼(ID, Customer_ID, Name, SSN)을 제거하여 모델 과적합(Overfitting) 방지.
파생변수 생성 (Feature Engineering): 도메인 지식을 바탕으로 총 16개의 파생변수를 생성.

비율 변수: Debt_to_Income, EMI_Load_Ratio, Cash_Flow_Ratio, Invest_Ratio 등
교호작용 변수: Age_x_History, Util_x_Inquiry, EMI_x_Delay, Delay_Risk_Score, Risk_Score 등
복합 변수: Loan_Type_Count (대출 유형 수 직접 파싱)


로그 변환: 우편향 분포를 가진 금융 수치 칼럼(Annual_Income, Outstanding_Debt 등)에 log1p 변환 적용.
이상치 클리핑: 1~99 퍼센타일 기준으로 이상치를 클리핑하여 모델 안정성 확보.
Encoding:

범주형 변수(Occupation, Credit_Mix, Payment_Behaviour 등)에 LabelEncoder 적용.
Month 변수는 순환성(cyclicality)을 보존하기 위해 Sin/Cos 변환 적용.


## Feature Selection: mutual_info_classif를 활용해 타겟 변수와 상호정보량(MI Score) > 0인 피처만 선별, 불필요한 노이즈 제거.
Scaling: TabNet의 특성에 맞게 수치형 피처에 QuantileTransformer(output_distribution='normal')를 적용하여 비선형 정규분포화.
Data Splitting: stratify=y 설정으로 신용 등급 클래스 비율을 유지하며 Train/Valid 세트 분할 (8:2).


## Exploratory Data Analysis (EDA) & Interpretation
데이터의 특성을 파악하고 신용 등급(Credit Score)에 영향을 미치는 주요 변수를 식별하기 위해 심층 분석을 수행했습니다.

### 1. 금융 수치 변수 분포 분석
히스토그램(KDE 포함)으로 Annual_Income, Monthly_Inhand_Salary, Outstanding_Debt, Credit_Utilization_Ratio, Total_EMI_per_month 등 금융 관련 수치 변수의 분포를 시각화했습니다.

분석 결과: 신용카드 사용 비율(Credit_Utilization_Ratio)을 제외한 대부분의 금융 변수는 분포가 고르지 않고 특정 구간에 집중되거나 극단값(이상치)이 존재하는 것을 확인했습니다.
인사이트: 도메인 특성상 자산·소득 데이터는 심하게 우편향(Right-skewed)되므로, 로그 변환 및 이상치 클리핑 전처리가 필수적임을 확인했습니다.
<img width="1487" height="990" alt="image" src="https://github.com/user-attachments/assets/8f089427-17f9-487a-ae41-1e834c4d47c0" />

### 2. 범주형 변수 분포 분석
Occupation, Credit_Mix, Payment_Behaviour, Payment_of_Min_Amount의 빈도(Frequency)를 Count Plot으로 시각화했습니다.
<img width="1489" height="1189" alt="image" src="https://github.com/user-attachments/assets/13922646-b121-4f65-875b-f5e9db5399e4" />

분석 결과: 직업군, 신용 조합 유형, 납부 습관이 각각 다양하게 분포하며, 특정 카테고리에 편중된 양상을 보였습니다.
인사이트: 직업 및 납부 행태는 개인의 신용 등급과 밀접하게 연관될 것으로 추정되어, 인코딩 후 모델의 주요 입력 피처로 활용했습니다.

### 3. 변수 간 상관관계 분석 (Correlation Heatmap)
수치형 변수 전체에 대한 상관관계 히트맵을 분석했습니다.
<img width="1087" height="1007" alt="image" src="https://github.com/user-attachments/assets/4eaee1e5-5124-4167-afd0-b8730fc82682" />
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/4665fb2b-ab20-43f0-93c8-c7da46886a76" />

분석 결과:

대부분의 변수는 약한~중간 수준의 상관관계를 보이며, 일부 금융 변수 간에는 강한 양(+) 또는 음(-)의 상관관계가 존재했습니다.
Outstanding_Debt ↔ Num_of_Loan: 상관계수 ≈ 0.64 (대출 건수가 많을수록 미상환 부채 증가)
Credit_History_Age ↔ Outstanding_Debt: 상관계수 ≈ -0.63 (신용 이력이 길수록 미상환 부채 감소)
소득 관련 변수(Annual_Income, Monthly_Inhand_Salary)끼리는 매우 강한 양의 상관관계



### 4. 신용 등급별 부채 수준 분석 (Credit Mix × Outstanding Debt)
Credit_Mix 유형에 따른 Outstanding_Debt 분포를 Box Plot으로 비교했습니다.

분석 결과:

Credit_Mix 0 (Poor): 부채가 전반적으로 많고, 개인 간 편차도 큰 패턴
Credit_Mix 1 (Standard): 부채가 적고 비교적 균일한 분포
Credit_Mix 2 (Good): 평균적으로 중간 수준이나 일부 고부채 보유자 존재


인사이트: 신용 조합 유형(Credit_Mix)이 달라질수록 부채 수준의 분포가 뚜렷하게 달라지므로, 신용 등급 예측에 있어 핵심적인 피처임을 확인했습니다.


### Modeling: Feature Engineering → TabNet Pre-training → Ensemble
모델링 파이프라인 개요
파생변수 생성 (16개)
    ↓
로그 변환 & 이상치 클리핑
    ↓
범주형 인코딩 & 순환 인코딩 (Month)
    ↓
Mutual Information 기반 Feature Selection
    ↓
QuantileTransformer (비선형 정규화)
    ↓
TabNet 비지도 사전학습 (Self-supervised Pre-training)
    ↓
TabNet 지도학습 Fine-tuning
    +
LightGBM 학습
    ↓
소프트 보팅 앙상블 (가중치 Grid Search 최적화)
모델 선택 이유
모델선택 이유TabNet어텐션 기반 피처 선택으로 표 형식 데이터에 특화된 딥러닝 모델. 희소성(Sparsity) 정규화를 통해 과적합 방지 및 해석 가능성 확보LightGBM리프 중심 분할 전략으로 빠른 학습과 높은 예측 성능. TabNet과 상호보완적 특성으로 앙상블 효과 극대화
개선 사항 (Improvements)

### TabNet 비지도 사전학습 (Self-supervised Pre-training)

TabNetPretrainer로 데이터의 80%를 마스킹 후 복구하는 방식으로 사전학습 수행.
지도학습 Fine-tuning 시 사전학습 가중치(from_unsupervised)를 로드하여 초기 표현 학습 품질 향상.


### TabNet 하이퍼파라미터 조정 (과적합 억제)

n_d, n_a: 64 → 32 축소
n_steps: 6 → 5 축소
lambda_sparse: 1e-3 → 1e-2 대폭 강화 (희소성 증가)
옵티마이저: NAdam → AdamW (가중치 감쇠 효과)
스케줄러: CosineAnnealingLR 적용으로 학습률 안정적 감소


### LightGBM 학습

n_estimators=2000, class_weight='balanced'로 클래스 불균형 대응.
reg_alpha, reg_lambda 정규화 항 설정으로 일반화 성능 확보.


### 소프트 보팅 앙상블 가중치 자동 탐색

TabNet과 LightGBM의 예측 확률을 혼합하는 최적 가중치(w)를 0.0~1.0 구간에서 Grid Search로 자동 탐색.
Validation Accuracy 기준으로 최적 가중치를 선정하여 최종 예측에 활용.




### Validation Score
모델Val AccuracyMacro F1TabNet 단독--LightGBM 단독--최적 앙상블 (최종)--

최종 모델: TabNet + LightGBM 소프트 보팅 앙상블 (최적 가중치 Grid Search 선정)


### Strategic Insights & Recommendations
모델 분석과 EDA 결과를 바탕으로 신용 등급 예측의 핵심 비즈니스 인사이트를 도출했습니다.
1. 부채 관리 지표가 신용 등급의 핵심 결정 요인
Outstanding_Debt와 Num_of_Loan의 강한 양의 상관관계(≈0.64)에서 확인되듯, 부채 규모와 대출 건수는 신용 등급 하락의 핵심 위험 신호입니다. 부채 상환 이력 관리와 대출 건수 최소화가 신용 등급 유지의 기본 전략이 되어야 합니다.
2. 신용 이력의 장기 관리가 부채 위험을 상쇄
Credit_History_Age와 Outstanding_Debt의 음의 상관관계(≈-0.63)는 장기 신용 이력이 부채 위험을 완화하는 효과가 있음을 시사합니다. 젊은 연령층의 조기 신용카드 발급 및 성실한 납부 이력 구축이 장기적으로 신용 등급 방어에 유리합니다.
3. 납부 습관 및 신용 조합이 등급 차별화의 핵심
Credit_Mix에 따른 부채 수준 분포 차이에서 확인되듯, 다양한 금융 상품을 균형 있게 보유하고 최소 납부금액(Minimum Amount)을 성실히 납부하는 습관이 신용 등급 상위 유지의 핵심 요인입니다.

### Project Conclusion
본 프로젝트는 파생변수 생성 → Mutual Information 기반 피처 선택 → TabNet 비지도 사전학습 → 지도학습 Fine-tuning → LightGBM 앙상블로 이어지는 고도화된 딥러닝 모델링 파이프라인을 구축하여 신용 등급 분류 성능을 향상시켰습니다.
단순한 MLP 기반 베이스라인에서 출발하여, 도메인 지식 기반의 피처 엔지니어링과 TabNet의 어텐션 메커니즘, LightGBM의 앙상블 효과를 결합함으로써 예측 신뢰도를 단계적으로 개선하였으며, EDA를 통해 데이터 기반의 구체적인 신용 관리 인사이트를 도출하는 성과를 거두었습니다.
