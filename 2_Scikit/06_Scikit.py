from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# GridSearchCV
# - 교체 검증과 최적 하이퍼 파라미터 튜닝을 한 번에
# - 사이킷런은 GridSearchCV를 이용해 Classifier나 Regressor와 같은 알고리즘에 사용되는 하이퍼 파라미터를 순차적으로 입력하면서
#   편리하게 최적의 파라미터를 도출할 수 있는 방안을 제공한다
# grid_parameters = {'max_depth': [1,2,3], 'min_sampels_split':[2,3] }
# 순번 max_depth min_samples_split
# 1     1           2
# 2     1           3
# 3     2           2
# 4     2           3
# 5     3           2
# 6     4           3
# cv 세트가 3이면 -> 파라미터 순차 적용 횟수(6) x cv 셋트수(3) = 학습/검증 총 수행횟수(18)


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()

prameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}
grid_dtree = GridSearchCV(dtree, param_grid=prameters, cv=3, refit=True, return_train_score=True)
# refit -> 학습하고 검증하면서 최적의 하이퍼 파리미터를 찾으면 학습을 시켜 버린다
grid_dtree.fit(X_train, y_train)

score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df[
    ['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']
])

pred = grid_dtree.predict(X_test)
print('최적 파라미터 : ', grid_dtree.best_params_)
print('최고 정확도 : ',accuracy_score(y_test, pred))

estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
print('최고 정확도 : ',accuracy_score(y_test, pred))
