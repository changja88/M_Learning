from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

ftr_df = iris_df.iloc[:, :-1]
tgt_df = iris_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(ftr_df, tgt_df, test_size=0.3, random_state=121)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(f'정확도 : ', accuracy_score(y_test, pred))
