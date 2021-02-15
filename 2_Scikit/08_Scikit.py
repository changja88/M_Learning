from sklearn.datasets import load_iris
import pandas as pd

# 피처 스케일링
# - 표준화는 데이터의 피처 각각이 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환하는 것을 의미
#   -> 정규분포 (에쁜 산모양) 으로 만들어 준다
# - 정규화는 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해주는 개념이다
#   -> 연봉 100 ~100000 이걸 0에서 1사이로 표현 해준다

# 사이킷런 피처 스케일링 지원
# - StandardScaler : 정규 분포로 만들어 준다
# - MinMaxSclaer : 데이터 값을 0 과 1 사이의 범위 값으로 변환한다 (음수 값이 있으면 -1 에서 1값으로 변환)
from sklearn.preprocessing import StandardScaler, MinMaxScaler


iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())

print('feature 들의 분산 값')
print(iris_df.var())

print('######################################################################')
scaler = StandardScaler()
scaler.fit(iris_df)
iris_scacled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scacled, columns=iris.feature_names)
print('feature 들의 평균 값')
print(iris_df_scaled.mean())

print('feature 들의 분산 값')
print(iris_df_scaled.var())

print('######################################################################')
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scacled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scacled, columns=iris.feature_names)
print('feature 들의 최소 값')
print(iris_df_scaled.min())

print('feature 들의 최대 값')
print(iris_df_scaled.max())
