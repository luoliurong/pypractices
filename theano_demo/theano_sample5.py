import theano.tensor as T
from theano import function
from theano import shared
import numpy

"""
demo for how to define a function with internal state with Theano
"""

x = T.dmatrix('x')
y = shared(numpy.array([[4,5,6]]))
z = x+y
f = function(inputs=[x], outputs=[z])
print("Original shared value:", y.get_value())
print("Original function evaluation result: ", f([[1,2,3]]))

y.set_value(numpy.array([[5,6,7]]))

print("Shared value now is: ", y.get_value())
print("function evaluation result now is: ", f([[1,2,3]]))

