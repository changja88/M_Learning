import numpy as np

# ndarray 편리하게 생성하기
sequence_array = np.arange(10)
print(sequence_array)

zero_array = np.zeros((3, 2), dtype='int32')
print(zero_array)

one_array = np.ones((3, 2))
print(one_array)

array1 = np.arange(10)
array2 = array1.reshape(2, 5)
print(array2)
array3 = array1.reshape(5, 2)
print(array3)

# reshpae -1 -> 가변적이라는 뜻
# (-1, 1), (-1,) 형태로 자주 사용
# -1 은 반듯이 한개만 넣어줘야 한다 
array2 = array1.reshape(-1, 1)
print(array2)
array3 = array2.reshape(-1)
print(array3)
