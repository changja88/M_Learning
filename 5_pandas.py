import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)

# Series -> 1개의 컬럼값으로만 구성된 1차원 데이터 셋
# DatFrame -> 2차원 데이터 셋

titanic_df = pd.read_csv(
    'Data/Titanic/titanic_train.csv',
)

head = titanic_df.head(5)  # 기본이 5개
print(head)

dic1 = {
    'Name': ['a', 'b', 'c', 'd'],
    'Year': [2011, 2012, 2013, 2014],
    'Gender': ['M', 'F', 'M', 'F']
}

data_df = pd.DataFrame(dic1)
print(data_df)

data_df = pd.DataFrame(dic1, columns=['Name', 'Year', 'Gender', 'Age'])
print(data_df)

data_df = pd.DataFrame(dic1, index=['one', 'two', 'three', 'four'])
print(data_df)

print(data_df.columns)
print(data_df.index)
print(data_df.index.values)

print('######################################################################')
series = titanic_df['Name']
print(series.head(3))

filtered_df = titanic_df[['Name', 'Age']]
print(filtered_df.head(3))

filtered_df = titanic_df[['Name']]  # 한개이지만 리스트로 전달을 하면 DataFrame으로 나온다
print(filtered_df.head(3))
print(type(filtered_df.head(3)))

print(filtered_df.shape)  # 자동으로 붙여주는 인덱스는 컬럼 카운트에 포함 되지 않는다

print('######################################################################')
print(titanic_df.info())  # 메타 데이터 반환

print('######################################################################')
print(titanic_df.describe())  # 평균, 표주편차, 4분위 분포도를 제공한다

print('######################################################################')
print(titanic_df['Pclass'].value_counts())
# 동일한 개별 데이터 값이 몇건이 있는지 정보를 제공
# series 객체에서만 호출 될 수 있으므로 반드시 DataFrame을 단일 컬럼으로 입력하여 Series로 변환한 뒤 호출한다

print('######################################################################')
sorted = titanic_df.sort_values(by='Pclass', ascending=True)
print(sorted)
sorted = titanic_df[['Name', 'Age']].sort_values(by='Age')
print(sorted)
sorted = titanic_df[['Name', 'Age', 'Pclass']].sort_values(by=['Pclass', 'Age'])
print(sorted)
