from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# 오차행렬
# - 오차행렬은 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 에측 오류가 발생하고 있는지를 함꼐 나타내느 지표
#          예측 클래스(Preidicted Class)
#             (0)         (1)
#   (0)     정답(TN)  |  실패(FP)
# 실제클래스 -------------------------
#   (1)     실패(FN)  |  정답(TP)

# - 정확도 : 예측 결과와 실제 값이 동일한 건수/ 전체 데이터 수 = (TN + TP) / (TN + FP + FN + TP)
# - 정밀도 : TP / (FP + TP) -> Precision_score()
#   - 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
# - 재현율 : TP / (FN + TP) -> recall_score()
#   - 실제 값이 Positive인 대상 중에 예과 실제 값이 Positive로 일치한 데이터의 비율

# - TN -> 거짓을 맞췄다
# - TP  -> 진실을 맞췄다
# - FP -> 진실로 틀렸다 (진실로 찍어서 틀렸다)
# - FN -> 거짓으로 틀렸다 (거짓으로 찍어서 틀렸다)

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df


# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df


# 레이블 인코딩 수행.
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


titanic_df = pd.read_csv('../Data/Titanic/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)

X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

print('######################################################################')


class MyDummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        # 단순히 Sex가 1이면 0 그렇지 않으면 1로 예측함
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1

        return pred


myclif = MyDummyClassifier()
myclif.fit(X_train, y_test)
mypredictions = myclif.predict(X_test)

def get_clf_eval(y_test, pred):
    confution = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차 행렬')
    print(confution)
    print(f'정확도 : {accuracy} 정밀도 : {precision} 재현율 : {recall}')

get_clf_eval(y_test, mypredictions)