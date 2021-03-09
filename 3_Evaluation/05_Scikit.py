from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_auc_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, Binarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diaetes_data = pd.read_csv('../Data/Pima_Indian/diabetes.csv')


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confution = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차 행렬')
    print(confution)
    print(f'정확도 : {accuracy} 정밀도 : {precision} 재현율 : {recall}')

    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print(f'F1 {f1}, AUC:{roc_auc}')


def precision_recall_curve_plot(y_test, pred_proba_c1, threasholds):
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


X = diaetes_data.iloc[:, :-1]
y = diaetes_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)

print('데이터 전처리 전######################################################################\n')
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, pred, pred_proba)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
precisions, recalls, threasholds = precision_recall_curve(y_test, pred_proba_c1)
precision_recall_curve_plot(y_test, pred_proba_c1, threasholds)


# precision_recall_curve_plot이 이상하게 그려지는 원인을 찾음 -> 0인 데이터가 많다
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
total_count = diaetes_data['Glucose'].count()
for feature in zero_features:
    zero_count = diaetes_data[diaetes_data[feature] == 0][feature].count()

# 0값을 평균 값으로 다 바꿔 준다
diaetes_data[zero_features] = diaetes_data[zero_features].replace(0, diaetes_data[zero_features].mean())

scaler = StandardScaler()
X_caled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_caled, y, test_size=0.2, random_state=156, stratify=y)

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)

print('데이터 전처리 후 ######################################################################\n')
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
print(pred_proba)
get_clf_eval(y_test, pred, pred_proba)

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
precisions, recalls, threasholds = precision_recall_curve(y_test, pred_proba_c1)
precision_recall_curve_plot(y_test, pred_proba_c1, threasholds)

print('임계값 조정 ######################################################################\n')

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임계값 : ', custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)

threasholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds=threasholds)