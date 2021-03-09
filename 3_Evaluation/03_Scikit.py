from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 업무에 따른 재현율과 정밀도의 상대적 중요도
# - 재현율이 상대적으로 더 중요한 지표인 경우에는 실제 positive 양성인 데이터 예측을 Negativie로 잘못 판단
#   하게 되면 업무상 큰 영향이 발생하는 경우 -> 암 진단, 금융 사기 판별
# - 정밀도가 상대적으로 더 중요한 지표인 경우에는 실제 Negative 음성인 데이터 예측을 Positive로 잘못 판단
#   하게 되면 업무상 큰 영향이 발생하는 경우 -> 스팸 메일

# 정밀도/재현율 트레이드 오프
# - 분류하려는 업무의 특성상 정도 또는 재현율이 특별히 강조돼야 할 경우 분류의 결정 임계값(Threshold)를 조정해
#   정밀도 또는 재현율의 수치를 높일 수 있다
# - 하지만 정밀도와 재현율은 상호 보완적인 평가 지표이기 때문에 어느 한쪽을 강제로 높이면 다른 하나의 수치는 떨어지기 쉽다
#   이를 정밀도/재현율 트레이드 오프라고 한다

# -> 분류 결정 임계값이 낮아질 수록 Positive로 예측할 확률이 높아짐 -> 재현율 증가
# -> predict_proba() 메소드는 분류 결정 예측 확률을 반환한다
#    이를 이용하면 임의로 분류 결정 임계값을 조정하면서 예측 확률을 변경할수 있다
# -> 정밀도 재현율 곡선
#   - precision_recall_curve() 함수를 통해 임계값에 따른 정밀도 재현율의 변화값을 제공한다


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


def get_clf_eval(y_test, pred):
    confution = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차 행렬')
    print(confution)
    print(f'정확도 : {accuracy} 정밀도 : {precision} 재현율 : {recall}')


titanic_df = pd.read_csv('../Data/Titanic/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)

X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

print('######################################################################1')

myclif = LogisticRegression()
myclif.fit(X_train, y_train)

pred_proba = myclif.predict_proba(X_test)
mypredictions = myclif.predict(X_test)

print(f'pred_proba : {pred_proba.shape}')
print(f'pred_proba 샘플 : {pred_proba[:3]}')

print('######################################################################2')

X = [[1, -1, 2],
     [2, 0, 0],
     [0, 1.1, 1.2]]
# threshold 기준 값보다 같거나 작으면 0을 크면 1을 반환
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

print('######################################################################3')
custom_threashold = 0.5
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)
binarizer = Binarizer(threshold=custom_threashold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)

print('######################################################################4')
pred_proba_class1 = myclif.predict_proba(X_test)[:, 1]
precisions, recalls, threasholds = precision_recall_curve(y_test, pred_proba_class1)
print(precisions[:5])
print(recalls[:5])
print(threasholds[:5])

print('######################################################################5')


def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, custom_threashold = precision_recall_curve(y_test, pred_proba_c1)

    plt.figure(figsize=(8, 6))
    threshold_boundary = threasholds.shape[0]
    plt.plot(threasholds, precisions[0: threshold_boundary], linestyle='--', label='precision')
    plt.plot(threasholds, recalls[0: threshold_boundary], label='recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threashold value')
    plt.ylabel('Precision and Recall values')
    plt.legend()
    plt.grid()
    plt.show()


precision_recall_curve_plot(y_test, myclif.predict_proba(X_test)[:, 1])
