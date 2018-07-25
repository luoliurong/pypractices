import numpy
import theano.tensor as T
from theano import function

"""
demo for how to define a function with vectors in Theano.
"""

a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')

e = (a+b-c)*d

# defin a function named f which takes a,b,c,d as input and produces e as output.
f = function([a,b,c,d],e)

a_data = numpy.array([[1,1],[1,1]])
b_data = numpy.array([[2,2],[2,2]])
c_data = numpy.array([[5,5],[5,5]])
d_data = numpy.array([[3,3],[3,3]])

print("Expected: ", (a_data + b_data-c_data)*d_data)
print("Via Theano: ", f(a_data, b_data, c_data, d_data))


