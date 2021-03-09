import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 분류 성능 평가 지표
# - 정확도 (Acuuracy)
# - 오차행렬 (Confusion Matrix)
# - 정밀도 (Percision)
# - 재현율 (Recall)
# - F1 스코어
# - ROC AUC

# 정확도
# - 직관적으로 모델 예측 성능을 나타내는 펴가 지표, 하지만 이진 부뉼의 경우 데이터의 구성에 따라 ML 모델의 성능을 왜곡 할 수 있기
#   때문에 정확도 수치 하나만 가지고 성능을 평가하지 않는다
# - 특히 정확도는 불균형한 레이블 값 분포에서 ML 모델의 성능을 판단할 경우, 적합한 평가 지표가 아니다
# - 예를 들어서 사기 카드 검출은 다 정상이다라고만 해도 90퍼센트를 얻을 수 있다


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

print('MyDummyClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, mypredictions)))
