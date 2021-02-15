# 사이킷
# - 파이썬 기반의 다른 머신러닝 패키지도 사이킷런 스타일의 API를 지향할 정도로 쉽고 가장 파이썬 스럽다
# - 머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API를 제공한다
# - 오랜 기간 실전 환경에서 검증됐으며, 매우 많은 환경에서 사용되는 성숙한 라이브러리 이다
# - 주로 Numpy, Scipy 기반 위에서 구축된 라이브러리 이다

# 머신 러닝을 위한 용어 속성
# - 피처, 속성?
#   - 피처는 데이터 세트의 일반 속성이다
#   - 머신러닝은 2차원 이상의 다차원 데이터에서도 많이 사용되므로 타겟값을 제외한 나머지 속성을 모두 피처로 지칭
# - 레이블, 클래스, 타겟(값), 결정(값)
#   - 다 똑같다

# Estimator (최상위)
#   - Classifier
#       - DecisionTreeClassifier
#       - RandomForestClassifier
#       - GrdientBoostingClassifier
#       - GaussianNB
#       - SVC
#   - Regressor
#       - LinearRegression
#       - Ridge
#       - Lasso
#       - RandomForestRegressor
#       - GradientBosstingRegressor

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

iris = load_iris()  # 파이썬 딕셔너리랑 비슷한 형태이다
iris_data = iris.data

iris_label = iris.target  # 결정 값
print('값 : ', iris_label)
print('명 : ', iris.target_names)
print('##############################################')

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))

print('##############################################')
X_train, X_test, y_train, y_test = train_test_split(
    iris_data, iris_label, test_size=0.2, random_state=11
)
# 인자 설명
# 전체 데이터, 결정값, 테스트 사이즈, random_state 수행할때 마다 같은게 랜덤으로 걸리도록 하는 시드넘버

df_clf = DecisionTreeClassifier(random_state=11)  # 결정추리 분류기 생성
df_clf.fit(X_train, y_train)  # 학습 시작

pred = df_clf.predict(X_test)  # 예측 시작
print(pred)

print('##############################################')
print(f'예측 정확도 {accuracy_score(y_test, pred)}')
