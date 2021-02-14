import pandas as pd
import numpy as np


# DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환

col_name1 = ['col1']
list1 = [1, 2, 3]
array1 = np.array((list1))

df_list1 = pd.DataFrame(list1, columns=col_name1)
print(df_list1)  # 1차원 리스트로 만듬

df_array1 = pd.DataFrame(array1, columns=col_name1)
print(df_list1)  # 1차원 ndarray로 만듬

print('######################################################################')
col_name = ['col1', 'col2', 'col3']
list = [
    [1, 2, 3],
    [11, 22, 33]
]
array = np.array(list)
print(array.shape)

df_list = pd.DataFrame(list, columns=col_name)
print(df_list)

df_array = pd.DataFrame(array, columns=col_name)
print(df_array)

dict = {'col1': [1, 11], 'col2': [2, 22], 'col3': [3, 33]}
df_dict = pd.DataFrame(dict)
print(df_dict)

print('######################################################################')
array = df_dict.values
print('type: ', type(array), 'shape :', array.shape)

list = df_dict.values.tolist()
print(list)

dict = df_dict.to_dict('list')
print(dict)

print('######################################################################')
# DataFrame 의 컬럽 데이터 셋 Access

titanic_df = pd.read_csv(
    '../Data/Titanic/titanic_train.csv',
)
titanic_df['Age_0'] = 0
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age'] * 10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
print(titanic_df.head(3))

print('######################################################################')
# DataFrame 데이터 삭제
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
# drop 시그니처
# inplace
#   True -> 원본 교체
#   False -> 변경된 dataFrame을 반환
# axis
#   0 -> 가로(로우) 삭제
#   1 -> 세로(컬럼) 삭제

print('######################################################################')
# index
# -> 기존에 있던 인덱스는 index라는 컬럼으로 바꿔 버린다
# -> 새롭게 인덱스를 만든다titanic_df.reset_index(inplace=False)


