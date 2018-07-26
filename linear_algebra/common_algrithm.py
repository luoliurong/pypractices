"""
here are some common algrithms in liner algebra
1. scipy.linalg contains all the functions in numpy.linalg, as well as other more advanced ones not contained in numpy.linalg
2. scipy is faster
"""
import numpy as np
import scipy as sp
from scipy import linalg

def inverse_number(string):
    """
    求一个排列的逆序数
    """
    ans = 0
    for i in range(len(string)):
        for j in range(i):
            if string[j] > string[i]:
                ans+=1
    
    return ans

#print(inverse_number(input("please input the number:")))


A = sp.mat('[1 2;3 4]')
print(A)
print(A.I)
print(A.A1)

B = sp.mat('[[5], [6]]')
print(B.T)

C = A*B.T
print(C)

a = np.array([[1,2],[3,4]])
b = np.array([[5],[6]])
s = sp.linalg.inv(a).dot(b) #this is slow performance
print(s)

s2 = np.linalg.solve(a, b)
print(s2)