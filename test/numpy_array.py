import numpy as np
from pprint import pp

maxint = 10
a = np.array([[0, 3, 0, 1], [0, 1, 5, 1], [0, 0, 0, 0], [1, 3, 4, 6]])
unique, counts = np.unique(a, return_counts=True)
s = np.sum(a)
r = np.arange(maxint)
h,r = np.histogram(a, bins = r, density=False)
hd,rd = np.histogram(a, bins = r, density=True)
cs = np.cumsum(h)
csd = np.divide(cs,s)

b = np.copy(a)
with np.nditer(b, op_flags=['readwrite']) as it:
   for x in it:
       x[...] = cs[x[...]]


print("array: \n",a)
print("sum: ",s)
print("hist: ",h)
print("hist sum: ",np.sum(h))
print("hist divide: ",csd)
print("hist divide sum: ",h/np.sum(h))
print("hist density: ",hd)
print("hist cum sum: ",cs)
print("hist cum sum divide: ",cs/s)
print("brray: \n",b)
b = b/s*maxint
print("brray divide: \n",b)
b = np.round(b).astype(int)
print("brray divide: \n",b)
