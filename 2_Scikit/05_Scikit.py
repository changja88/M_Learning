from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# 교체 검증을 보다 간편하게
# - cross_val_score() 함수로 폴드 세트 추출, 학습/예측, 평가를 한번에 수행
# - cross_val_score(
# estimator, X, y=None, scoring=None, cv=None, n_job=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs'
# )


iris_data = load_iris()

dt_clf = DecisionTreeClassifier(random_state=156)
data = iris_data.data
label = iris_data.target

scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)
# 교차 검증 세트는 3개
print('교차 검증별 정확도', np.round(scores, 4))
print('평균 검증별 정확도', np.round(np.mean(scores), 4))
