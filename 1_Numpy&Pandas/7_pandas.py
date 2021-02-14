import pandas as pd


titanic_df = pd.read_csv(
    '../Data/Titanic/titanic_train.csv',
)

# 데이터 셀렉션 및 필터링
# [] -> 컬럼 기반 필터링 또는 불린 인덱싱 필터링 제공

# 명칭(Lable) -> 컬럼의 명칭을 기반으로 위치를 지정하는 방식 (컬럼 명 같이 명칭으로 열 위치를 지정하는 방식)
# 위치(Position) -> 행, 열 위치값으로 정수가 입렵된다
# ix[] -> 명칭 기반과 위치 기반 인덱싱을 함께 제공럼 (사라질 예정)
# loc[] -> 명칭 기반 인덱싱
# iloc[] -> 위치기반 인덱싱

# 불린 인덱싱 -> 조건식에 따른 필터링 제공

print('######################################################################')
print(titanic_df['Pclass'].head(3))
print(titanic_df[['Survived', 'Pclass']].head(3))

print(titanic_df[0:2])  # 허용은 해주지만 안하는게 좋다

print('######################################################################')
dic = {
    'Name': ['a', 'b', 'c', 'd'],
    'Year': [2011, 2012, 2013, 2014],
    'Gender': ['M', 'F', 'M', 'F']
}
data_df = pd.DataFrame(dic)
print(data_df.iloc[0, 0])  # iloc는 위치 기반 이기때문에 무조건 인자로 숫자가 온다
print(data_df.loc[0, 'Name'])

print(
    titanic_df[titanic_df['Age'] > 60]
)
print('######################################################################')
print(
    titanic_df[['Name', 'Age']][titanic_df['Age'] > 60]
)
print('######################################################################')
print(
    titanic_df[
        (titanic_df['Age'] > 60) & (titanic_df['Pclass'] == 1) & (titanic_df['Sex'] == 'female')
        ]
)
