from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# F1 Score
# - 정밀도와 재현율을 결합한 지표이다. F1 스코어는 정밀도와 재현율이 어느 한쪽으로 치우치지 않는 수치를 나타낼 때
#   상대적으로 높은 값을 가진다

# ROC 곡선과 AUC(Area Under Curve)
# - Receiver Operation Characticstic Curve 과 이에 기반한 AUC 스코어는 이진 분류의 예측 성능 측정에서
#   중요하게 사용되는 지표이다. 일반적으로 의한 분야에서 많이 사용되지만, 머신러닝의 이진 분류 모델의 예측 성능을 판단하는
#   중요한 평가 지표이기도 하다
# - ROC 곡선은 FPR이 변할때 TPR이 어떻게 변하는지를 나타내는 곡선이다
# - AUC 곡선 의 값이 1에 가까울수록 좋은 수치이다

# TPR (True Positive Rate) -> 재현율, 민감도
# - 진짜를 진짜로 맞출 확률
# - TP / (FN + TP)

# FPR
# - 실제 음성을 잘못 예측한 비율
# - FP / (FP + TN)

def get_clf_eval(y_test, pred):
    confution = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1_score1 = f1_score(y_test, pred)
    print('오차 행렬')
    print(confution)
    print(f'정확도 : {accuracy} 정밀도 : {precision} 재현율 : {recall} F1 : {f1_score1}')


def roc_curve_plot(y_test, pred_proba_c1):
    fprs, tprs, threshold = roc_curve(y_test, pred_proba_c1)

    plt.plot(fprs, tprs, lable='ROC')
    plt.plot([0, 1], [0, 1], 'k--', lable='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabll('FPR(1 - Sensitivity')
    plt.ylable('TPR (Recall)')
    plt.legend()
    plt.show()

