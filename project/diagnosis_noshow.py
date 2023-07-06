import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 데이터 불러오기
path = './project/'
data = pd.read_csv(path + 'medical_noshow.csv')

# 데이터 분할
x = data[['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Gender']]
y = data[['No-show']]
print(x.head(5))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# 데이터 전처리
data = data[['No-show', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','Gender']]

# 범주형 데이터를 이진 변수로 변환
data['No-show'] = data['No-show'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['Hipertension'] = data['Hipertension'].map({'Yes': 1, 'No': 0})
data['Diabetes'] = data['Diabetes'].map({'Yes': 1, 'No': 0})
data['Alcoholism'] = data['Alcoholism'].map({'Yes': 1, 'No': 0})

data.fillna(-1, inplace=True)

# 상관관계 분석
correlation = data[['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'No-show']].corr()
print(correlation)


# # 다중 클래스 분류를 위한 Random Forest 모델 학습
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # 예측 및 평가
# y_pred = model.predict(X_test)
# print(confusion_matrix(y_test.values.ravel(), y_pred))
# print(classification_report(y_test.values.ravel(), y_pred))
