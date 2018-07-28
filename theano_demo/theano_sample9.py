import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

"""
demo for how to define a function with a random variable.
use case:
where we want to define a function having a random variable, for example, introducing minor corruptions in inputs.
"""

random = RandomStreams(seed = 42)

a = random.normal((1,3))
b = T.dmatrix('b')
f = a * b
g = function([b], f)
print("Invocation1: ", g(numpy.ones((1,3))))
print("Invocation2: ", g(numpy.ones((1,3))))
print("Invocation3: ", g(numpy.ones((1,3))))
