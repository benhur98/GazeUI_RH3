import numpy as np
a=np.load("train-data-{}.npy".format(input()))
while 1:
    print(a[int(input())][1])
