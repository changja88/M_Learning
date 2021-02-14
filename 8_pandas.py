import pandas as pd


titanic_df = pd.read_csv(
    'Data/Titanic/titanic_train.csv',
)

print(titanic_df.count())
print('######################################################################')
print(titanic_df[['Age', 'Fare']].mean(axis=0))
print(titanic_df[['Age', 'Fare']].sum(axis=0))
print(titanic_df[['Age', 'Fare']].count(axis=0))
# 기본 axis = 0 이고 세로로 aggregation을 한다

print('######################################################################')
titanic_groupby = titanic_df.groupby(by='Pclass')
print(titanic_groupby.count())
# groupby 를 하게되면 이전과는 다르게 인덱스에 이름이 생긴다(여기에서는 Pclass)


print('######################################################################')
a = titanic_df.groupby('Pclass')['Age'].agg([max, min])
print(a)
agg_format = {'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
b = titanic_df.groupby('Pclass').agg(agg_format)
print(b)