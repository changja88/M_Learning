import pandas as pd


titanic_df = pd.read_csv(
    'Data/Titanic/titanic_train.csv',
)


# 결손 데이터 처리하기
# isna() -> True,False 값 반환
# fillna() -> Missing 데이터를 인자로 주어진 값으로 대체한다

print('######################################################################')

print(titanic_df.isna().head(3))
print(titanic_df.isna().sum())

print('######################################################################')
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
print(titanic_df.head(3))
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())