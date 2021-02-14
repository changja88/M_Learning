import pandas as pd


titanic_df = pd.read_csv(
    'Data/Titanic/titanic_train.csv',
)


# 판다스 apply lambda
def get_square(a):
    return a ** 2


lambda_square = lambda x: x ** 2
# lambda x (입력인자) : x ** 2 (입력 인자를 기반으로 한 계산식)
# 판다스의 경우 컬럼에 일괄적으로 데이터 가공을 하는 것이 속도 면에서 더 빠르나 복잡한 데이터 가공이 필요한 경우 어쩔 수 없이
# apply lambda 를 이용한다
print(get_square(3))
print(lambda_square(3))

print('##############################################')

titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x: len(x))
print(titanic_df['Name_len'])

print('##############################################')
titanic_df['Child_Adult'] = titanic_df['Age'].apply(
    lambda x: 'Child' if x < 15 else ('Adult' if x <= 60 else 'Elderly')
)
print(titanic_df['Child_Adult'])

print('##############################################')


def get_category(age):
    cat = ''
    if age <= 5:
        cat = 'Bay'
    elif age <= 12:
        cat = 'Child'
    elif age <= 18:
        cat = 'Teenager'
    return cat


titanic_df['Child_Adult'] = titanic_df['Age'].apply(
    lambda x: get_category(x)
)
titanic_df['Child_Adult']
