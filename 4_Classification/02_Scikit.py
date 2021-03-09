import warnings

import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import seaborn
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

dt_clf = DecisionTreeClassifier(random_state=156)
iris_data = load_iris()
X_train, X_teset, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

dt_clf.fit(X_train, y_train)

export_graphviz(
    dt_clf,
    out_file='tree.dot',
    class_names=iris_data.target_names,
    feature_names=iris_data.feature_names,
    impurity=True,
    filled=True
)

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# dot -Tpdf tree.dot -o graph1.pdf
# -> 닷을 이용해서 dot 파일을 pdf로 변환 시키면 볼수 있다

# 중요한 Feature 찾아내는 방법
print(f'Feature Importance : {np.round(dt_clf.feature_importances_,3)}')
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print(f'{name} {value}')

seaborn.barplot(x= dt_clf.feature_importances_, y=iris_data.feature_names)
plt.show()