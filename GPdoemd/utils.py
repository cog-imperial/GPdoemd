
import numpy as np

def expand_dims (a,axis):
    if isinstance(axis,int):
        return np.expand_dims(a,axis)
    b = a.copy()
    for ax in axis:
        b = np.expand_dims(b,ax)
    return b




