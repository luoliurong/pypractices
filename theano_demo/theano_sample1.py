import theano.tensor as T
from theano import function
"""
demo for how to define a function with scalars in Theano.
"""
# scalars must be defined before they can be used, and each scalar has a unique name.
a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')

f = ((a-b+c)*d)/e

# define the function with name g, which takes a,b,c,d,e as input and produce f as the output.
g = function([a,b,c,d,e], f)

# compute the result of the function g with the non-theano expression.
print("Expected: ((1-2+3)*4)/5.0 = ",((1-2+3)*4)/5.0 )
print("Via Theano: ((1-2+3)*4)/5.0 = ", g(1,2,3,4,5))
