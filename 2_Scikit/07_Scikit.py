from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# 데이터 전처리 (Preprocessing)
# - 데이터 클린징
# - 결손값 처리(Nall/Nan 처리) -> 머신 러닝은 null을 받지 않는다
# - 데이터 인코딩(레이블, 원-핫 인코딩) -> 머신 러닝은 숫자만 받아 들인다
# - 데이터 스케일링 -> 정규화, 표준화, 단위 맞추기...
# - 이상치 제거 -> 택도 없는 값 제거
# - Feature 선택, 추출 및 가공

# 데이터 인코딩
# - 머신러닝 알고리즘은 문자열 데이터 속성을 입력 받지 않으며 모든 데이터는 숫자형으로 표현되어야 한다
#   문자형 카테고리형 속성은 모두 숫자값으로 변환/인코딩 되어야 한다
# - 레이블 인코딩
#   - TV -> 0 냉장고 -> 1 문자열을 숫자 값으로 치환
#   - 숫자로 바꿔 버리면 숫자끼리 연관성이 생겨 버릴수도 있따 (티비, 냉장고는 아무 연관성이 없지만)
# - 원-핫 인코딩
#   - 위에 문제를 해결 하기 위해서 도입된 방식
#   - 피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 칼럼에만 1을 표시하고 나머지 컬럼에는 0을 표시 하는 방식
#   - 레이블 인코딩을 하고 -> 원-핫 인코딩을 한다
#   - TV 냉장고 믹서 선풍기 전자렌지 컴퓨터 가격
#   - 1   0    0    0     0      0   100
#   - 0   1    0    0     0      0   100

items = ['TV', '냉장고', '전제렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값 : ', labels)
print('인코딩 클래스 : ', encoder.classes_)
print('디코딩 원본 값 : ', encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))

print('######################################################################')
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

labels = labels.reshape(-1, 1)  # 2차원 데이터로 변환

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_lables = oh_encoder.transform(labels)

print(oh_lables.toarray())
print(oh_lables.shape)

print('######################################################################')
df = pd.DataFrame({'item': ['TV', '냉장고', '전제렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
print(df)
dummy = pd.get_dummies(df)
print(dummy)