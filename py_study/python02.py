import numpy as np

# 문제 1번
print("makit 'code' lab she's gone")

# 문제 2번
a = 10
b = 20
c = a + b
print('a의 값은', a)
print('b의 값은', b)
print('a와 b의 합은', c)

# 문제 3번
a = 10
b = 'makit'
print (a * 3)
print (b * 3)

# 문제 4번
a = ['메이킷', '우진', '시은']
print(a)
print(a[0])
print(a[1])
print(a[2])

# 문제 5번
a = ['메이킷', '우진', '제임스', '시은']
print([a[0], a[1]])
print('-------------------')
print(a[0:2])
print('-------------------')
print([a[1], a[2], a[3]])
print([a[2], a[3]])
print(a)

# 문제 6번
a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']
print(a + b)
print(b)

# 문제 7번 리스트를 붙이는 명령어 extend()
b.extend(a)
print(b)

# 문제 8번
a = np.array([[1, 2, 3], [4, 5, 6]])
print("Original :\n", a)

a_trainspose = np.transpose(a)
print('Transpose :\n', a_trainspose)

b = np.array([[1, 2, 3], [4, 5, 6]])
print("Original :\n", b)

b_reshape = np.reshape(a, (3, 2))
print('Reshape :\n', b_reshape)