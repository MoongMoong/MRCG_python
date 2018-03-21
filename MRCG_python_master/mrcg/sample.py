import numpy as np
from scipy import signal

l=[1,2,3]
print(sum(map(lambda x:x*x,l)))
print(np.exp(2))
a = [[1,2],[3,4]]
b = [[4,5],[6,7]]
c = np.multiply(a,b)
d = np.divide(2,a)
e = np.subtract(a,-1)
print(e)
# c = range(5)
# for i in c:
#     print(i)
# c = np.zeros([2,3])
# a = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])
# for i in [0,1,2]:
#     start = int(i*2)
#     b = a[:,start:start+6]
#     c[:,i] = np.sum(b,1)
# print(c)
# d = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
# print(np.sum((d),2))
# epsc =0.000001
# a =np.arange(1, 21+epsc, 1)
# b = [2**x for x in a ]
# print(b)
# d = np.reshape(d,[2,3])
# e,f = np.shape(d)
# print(e)
# print(f)
# # ff = np.multiply([0.0020,0.0025,0.0032,0.0040,0.0050,0.0063,0.0080,0.0100,0.0125,0.0160,0.0200,0.0250,0.0315,0.0400,0.0500,0.0630,0.0800,0.1000,0.1250,0.1600,0.2000,0.2500,0.3150,0.4000,0.5000,0.6300,0.8000,1.0000,1.2500],10000)
# # print(ff)
# w = 9
# hlen = np.floor(w / 2)
# w = 2 * hlen + 1
# win=np.arange(hlen, -(hlen+1), -1)
# print(win)
#
# b, a = signal.butter(8, 0.125)
# print(b)
# print(a)