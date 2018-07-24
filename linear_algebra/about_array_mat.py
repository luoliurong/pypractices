from numpy import * ####导入库函数
import numpy as np  ####以这种方式使用numpy的函数时,要以np开头

######################################################################
print('matrix 创建......')
##1. create matrix
array1 = array([1,2,3])
matrix1 = mat(array1)
print(matrix1)

#######create a zero matrix
zeroMat = mat(zeros((3,3)))
print(zeroMat)

matrix2 = mat(ones((2,4)))
print(matrix2)

matrix2 = mat(ones((2,4), dtype=int))
print(matrix2)

matrix3 = mat(random.rand(2,2))
print(matrix3)

matrix4 = mat(random.randint(10, size=(3,3)))
print(matrix4)

matrix5 = mat(random.randint(2,5,size=(2,5)))
print(matrix5)

##4X4对角矩阵
matrix6 = mat(eye(4,4, dtype=int))
print(matrix6)

##对角矩阵，各个值不一样
array2 = [1,2,3,4]
matrix7 = mat(diag(array2))
print(matrix7)

######################################################################
print('matrix 运算......')
##2, 常用的矩阵运算
#矩阵相乘
matrix1 = mat([1,2])
matrix2 = mat([[1],[2]])
matrix3 = matrix1 * matrix2
print(matrix3)

#矩阵对应元素相乘
matrix1 = mat([1,1])
matrix2 = mat([2,2])
matrix3 = multiply(matrix1, matrix2)
print(matrix3)

#矩阵点乘
matrix1 = mat([2,3])
matrix2 = matrix1*2
print(matrix2)

#矩阵求逆
matrix1 = mat(eye(2,3)*0.5)
print(matrix1)
matrix2 = matrix1.I
print(matrix2)

#矩阵转置
matrix1 = mat([[1,2],[3,4]])
matrix2 = matrix1.T
print(matrix2)

#计算每一行，列的相关值
matrix1 = mat([[1,1],[2,3],[4,2]])
#求列的和
colSum = matrix1.sum(axis=0)
print(colSum)
#求行的和
rowSum = matrix1.sum(axis=1)
print(rowSum)
#求某一行所有列的和
row1Sum = sum(matrix1[0,:])
print(row1Sum)
#矩阵中所有元素的最大值
maxVal = matrix1.max()
print(maxVal)
#求第二列的最大值
col2Max = max(matrix1[:,1])
print(col2Max)
#求第二行的最大值
row2Max = matrix1[1,:].max()
print(row2Max)
#求所有列的最大值
print('max of all columns')
colMax = np.max(matrix1, 0)
print(colMax)
#求所有行的最大值
rowMax = np.max(matrix1, 1)
print(rowMax)
#所有列的最大值在该列中对应的索引
colMaxIdx = np.argmax(matrix1, 0)
print(colMaxIdx)
#第二行中最大值对应在该行的索引
row2MaxIdx = np.argmax(matrix1[1,:])
print(row2MaxIdx)

#矩阵的分割和合并
array1 = [[1,2,3],[4,5,6],[7,8,9]]
matrix1 = mat(array1)
#分割出第二行以后的行和第二列以后的列的所有元素
matrix2 = matrix1[1:, 1:]
print(matrix1)
print(matrix2)
#合并矩阵
matrix1 = mat(ones((2,2)))
print(matrix1)
matrix2 = mat(eye(2))
print(matrix2)
#按列合并
print('按列合并')
matrix3 = vstack((matrix1, matrix2))
print(matrix3)
#按行合并，即行数不变，扩展列数
matrix4 = hstack((matrix1, matrix2))
print(matrix4)

######################################################################
print('矩阵和列表，数组的转换......')
#list
list = [[1],'hello',3]
#numpy数组中，同一个数组的所有元素必须为同一个类型
array1 = array([[2],[1]])
dim = array1.ndim
m,n = array1.shape
siz = array1.size
ty = array1.dtype
#矩阵与数组之间的转换
list = [[1,2],[3,2],[5,2]] #列表
array1 = array(list) #列表 - 数组
matrix1 = mat(list) #列表 - 矩阵
array2 = array(matrix1) #矩阵 - 数组
list2 = matrix1.tolist() #矩阵 - 列表
list3 = array2.tolist() #数组 - 列表
print(matrix1)

######################################################################
print('其他运算......')
matrix1 = mat([[1,2,3],[4,5,6]], dtype=int)
matrix2 = 2*matrix1
print(matrix2)
matrix3 = matrix1+matrix2
print(matrix3)
matrix2 = mat([[1,2],[3,4],[5,6]], dtype=int)
matrix4 = matrix1*matrix2
print(matrix4)
