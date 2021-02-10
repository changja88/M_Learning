import numpy as np

# 인덱싱
# 마이너스로 시작하면 뒤에서 부터 라는 뜻
# array1[0:3] -> 0번째 부터 2번째 까지

# 슬라이싱 -> 연속된 인덱스상의 ndarray 추출
# 팬시 인덱싱 -> 일정한 인덱싱 집합을 리스트 또는 ndarray 형재로 지정해서 인덱싱
# 불린 인덱싱 -> 특정 조건에 해당하는지 여부를 확인해서 인덱싱


array1 = np.arange(start=1, stop=10)
array2d = array1.reshape(3, 3)
print(array2d)

# 팬시 인덱싱
array3 = array2d[[0, 1], 2]
print(array3)

# 불린 인덱싱
boolean_index = array1 > 5  # 조건식 값이 반환이 된다
print(boolean_index)

array2 = array1[boolean_index] # 불린 리스트를 넣어주면 작동한다 
print(array2)

array2 = array1[array1 > 5]
print(array2)
