from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df['label'].value_counts())

print('##############################################')
kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'교차 검증 {n_iter}')
    print(f'학습 레이블 데이터 분포 : {label_train.value_counts()}')
    print(f'검증 레이블 데이터 분포 : {label_test.value_counts()}')

# -> 이렇게 나눠 주게 되면 망한다

print('##############################################')
skf = StratifiedKFold(n_splits=3)
n_iter = 0
for train_indx, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(f'교차 검증 {n_iter}')
    print(f'학습 레이블 데이터 분포 : {label_train.value_counts()}')
    print(f'검증 레이블 데이터 분포 : {label_test.value_counts()}')
# -> 이렇게 나눠 주면 테스트 셋들이 고르게 라벨들을 학습 할수 있다
