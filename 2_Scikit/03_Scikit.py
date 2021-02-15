from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# 교차 검증
# - 학습 데이터를 다시 분할하여 학습 데이터와 학습된 모델의 성능을 일치 평가하는 검증 데이터로 나눔
# - 학습 데이터 세트 -> 학습 데이터 세트 + 검증 데이터 세트
# - 모든 학습/검증 과정이 완료된 후 최종적으로 성능을 평가하기 위한 데이터 세트 -> 테스트 데이터 세트

# K 폴드 교차 검증
# - K가 5일 경우
#   - 학습 학습 학습 학습 검증
#   - 학습 학습 학습 검증 학습
#   - 학습 학습 검증 학습 학습
#   - 학습 검증 학습 학습 학습
#   - 검증 학습 학습 학습 학습
#   -> 교차 검증 최종 평가 = 평균 ( 평가[1,2,3,4,5] )

# Stratified K 폴드
# - 불균형한 (imbalanced) 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K 폴드 방식
# - 학습 데이터와 검증 데이터 세트가 가지는 레이블 분포도가 유사하도록 검증 데이터 추출
# - 예를 들어 위조 카드를 검출 학습을 한다면 위조 카드 갯수가 너무 적기 때문에 학습 데이터를 만들때 어떤 학습에는 위조 카드가 없는
#   경우가 많이 있을수 있다 -> 이를 균일하게 만들어 주는 방식

iris = load_iris()
features = iris.data
label = iris.target
dt_clif = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)
cv_accuracy = []
print(f'붓꽃 데이터 세트 크기 : {features.shape[0]}')

n_iter = 0
for train_index, test_index in kfold.split(features):
    # split -> train index, test index 를 리턴 해준다
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clif.fit(X_train, y_train)
    pred = dt_clif.predict(X_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print(f'{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기 : {train_size} 검증 데이터 크기 : {test_size} ')
    print(f'{n_iter}  검증 세트 인덱스 {test_index}')

    cv_accuracy.append(accuracy)

print(f'평균 검증 정확도 : ',np.mean(cv_accuracy))

