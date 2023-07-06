from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


path = './project/'
datasets = pd.read_csv(path + 'medical_noshow.csv')
datasets.info()
datasets.describe()

# 결측치
datasets.isnull().sum()
datasets.shape

# 데이터 전처리 
datasets['Gender'] = datasets['Gender'].apply( lambda x : 1 if x == 'M' else 0) # 여성은 0, 남성은 1로 변환
datasets['No-show'] = datasets['No-show'].apply( lambda x: 1 if x == 'Yes' else 0) # Noshow 1 , show 0
# datasets['No-show'] = datasets['No-show'].map({'No':0, 'Yes':1}) # No-show를 0과 1로 변환 Noshow 1, show 0
## 새 컬럼, 예약 시간만 포함하는 컬럼 생성
datasets['ScheduledDay'] = pd.to_datetime(datasets.ScheduledDay)
datasets['AppointmentDay'] = pd.to_datetime(datasets['AppointmentDay'])# 진료일의 날짜 부분 추출
datasets['ScheduleTime'] = datasets.ScheduledDay.dt.time
datasets['ScheduledDay'] = datasets.ScheduledDay.dt.normalize()
datasets['WaitingDays'] = (datasets['AppointmentDay'] - datasets['ScheduledDay']).dt.days # 예약일로부터 진료일까지의 일수 계산

datasets.info()
datasets.head(10)

df_age = datasets.query('Age < 0')
df_age
datasets.drop(df_age.index, inplace=True)
datasets['Handcap'] = np.where(datasets['Handcap']>0, 1, 0)

df_day = datasets.query('WaitingDays < 0')
datasets.drop(df_day.index, inplace=True)

datasets['ScheduledDayofWeek'] = pd.to_datetime(datasets['ScheduledDay']).dt.dayofweek
datasets['AppointmentDayofWeek'] = pd.to_datetime(datasets['AppointmentDay']).dt.dayofweek
datasets.info()
datasets.head(10)

new_col = {'Neighbourhood':'Neighborhood','Handcap':'Handicap'}
datasets.rename(columns = new_col , inplace = True)

# 지역 번호 부여
from sklearn.preprocessing import LabelEncoder

ob_col = list(datasets.dtypes[datasets.dtypes=='object'].index)
for col in ob_col:
    datasets['NeighborhoodNum'] = LabelEncoder().fit_transform(datasets['Neighborhood'].values)  # 지역에 따라 번호 부여

# 시간 형식 변경
datasets['ScheduleTime'] = pd.to_datetime(datasets['ScheduleTime'], format='%H:%M:%S')
datasets['ScheduleTime'] = datasets['ScheduleTime'].dt.hour.astype(float)

### 환자별로 몇번 No-show를 하였나를 측정 ###
datasets['Num_App_Missed'] = datasets.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())

# 필드 제외
exclude_fields = ['ScheduledDay', 'AppointmentDay', 'Neighborhood']

# Train-test 분할
x = datasets.drop(['No-show'] + exclude_fields, axis=1)
y = datasets['No-show']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# kfold
n_splits = 7
random_state = 72
kfold = KFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 기본 모델 정의
base_models = [
    ('lgbm', LGBMClassifier(n_estimators=2000, learning_rate=0.09, random_state=42)),
    ('logreg', LogisticRegression(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# 스태킹 분류기 초기화
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=RandomForestClassifier(random_state=42),
    cv=kfold
)

# 스태킹 모델 훈련
stacking_model.fit(x_train, y_train)

# 스태킹 모델 평가
train_predictions = stacking_model.predict(x_train)
train_accuracy = np.mean(train_predictions == y_train)
print("Train Accuracy:", train_accuracy)

score = cross_val_score(
    stacking_model,
    x, y,
    cv=kfold
)

print('Acc score:', score)
print('Cross Val Score:', round(np.mean(score), 4))
