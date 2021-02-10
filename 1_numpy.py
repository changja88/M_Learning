import numpy as np


array1 = np.array([1, 2, 3])
print('array1 tpye', type(array1))
print('array1 array 형태', array1.shape)
print('array1 tpye :', array1.dtype)

array2 = np.array(
    [
        [1, 2, 3],
        [2, 3, 4]
    ]
)
print('array2 tpye', type(array2))
print('array2 array 형태', array2.shape)

array3 = np.array(
    [
        [1, 2, 3]
    ]
)
print('array3 tpye', type(array3))
print('array3 array 형태', array3.shape)

demension = f'array 1차원: {array1.ndim}, array 2차원: {array2.ndim},array 3차원: {array3.ndim},'
print(demension)

list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2.dtype)  # arry2의 타입이 전부다 string으로 바뀐 (큰 걸로 바뀐다)

array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')  # 타입 변
print(array_float, array_float.dtype)

array2 = np.array(
    [[1, 2, 3],
     [2, 3, 4]]
)
print(array2.sum())
print(array2.sum(axis=0))  # 세로방향
print(array2.sum(axis=1))  # 가로방향
