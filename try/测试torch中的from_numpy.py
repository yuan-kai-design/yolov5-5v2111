from torch import from_numpy
from numpy import array
a=array([[1,2],
         [3,4]])
v=from_numpy(a)
print(v.dtype)
v=v.float().half()
print(v.dtype)
#
