import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score



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
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)


# 모델
model = LGBMClassifier(n_estimators=2000, learning_rate=0.1, random_state=42)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=100)
model.fit(x_train, y_train)


# 예측 및 평가
score = cross_val_score(
    model,
    x, y,
    cv = kfold
)

print('acc score : ', score,
      '\n cross_Val_Score : ', round(np.mean(score), 4))

# test_predictions = model.predict(x_test)
# test_accuracy = np.mean(test_predictions == y_test)
# print("Test Accuracy:", test_accuracy)

# Train Accuracy: 0.9106383460007238
# Test Accuracy: 0.7990952273241348

#  cross_Val_Score :  0.928


# 시각화
plt.figure(figsize=(12, 8))
corr_matrix = datasets.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('No-show HitMap')
plt.show()