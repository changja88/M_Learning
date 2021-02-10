import numpy as np


# np.sort() -> 원 행렬은 그대로 유지한 채 원 행렬의 정렬된 행렬을 반환
# ndarray.sort() -> 원 행렬 자체를 정렬한 형태로 변환하며 반환값은 none
# 기본적으로 둘다 오름 차순으로 정렬을 하고 내림차순 정렬을 하려면 [::-1] 해주면 된다

# agsort()
# -> sorting을 하는데 값으로 채우는게 아니라 index로 채운다
# -> [6, 5, 3] -> [2, 1, 0]이 나온다

# dot()
# -> np.dat(A,B) -> 행렬 A,B의 값을 구한다

# transpose(A)
# 전치 행렬

org_array = np.array([3, 1, 9, 5])
print(org_array)

sort_array1 = np.sort(org_array)
print(sort_array1)  # 원본은 변하지 않는다

sort_array2 = org_array.sort()
print(sort_array2)  # 원본이 변한다
print(sort_array1)

array2d = np.array([[8, 12],
                    [7, 1]])
sort_array_axis0 = np.sort(array2d, axis=0)
print(sort_array_axis0)
sort_array_axis0 = np.sort(array2d, axis=1)
print(sort_array_axis0)

org_array = np.array([3, 1, 9, 5])
print(np.sort(org_array))
sort_index = np.argsort(org_array)
print(sort_index)
sort_index_desc = np.argsort(org_array)[::-1]
print(sort_index_desc) # 내림 차순 index
