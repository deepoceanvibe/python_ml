import numpy as np
from sklearn.covariance import EllipticEnvelope 


outliers_data = np.array([
    -50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 100
])
print(outliers_data.shape)      # (21,)
outliers_data = outliers_data.reshape(-1, 1)        # 데이터 열 하나 더만들어주기
print(outliers_data.shape)      # (21, 1)

# EllipticEnvelope적용
outliers = EllipticEnvelope(contamination=.3)      # (0.3) 전체 데이터의 30% 이상치 탐지
outliers.fit(outliers_data)                        # 모델을 이상치 데이터에 맞춤
result = outliers.predict(outliers_data)           # 데이터 이상치 예측 (이상치는 -1, 정상치는 1)

print(result)
print(result.shape)



# 시각화
# import matplotlib.pyplot as plt
# plt.boxplot(outliers_data)
# plt.show()