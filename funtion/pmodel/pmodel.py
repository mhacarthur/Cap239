#P-model from Meneveau & Sreenevasan, 1987 & Malara et al., 2016
#Author: R.R.Rosa & N. Joshi
#Version: 1.6
#Date: 11/04/2018

import numpy as np
from matplotlib import pyplot


def pmodel(noValues=1024, p=0.4999, slope=[]):
    noOrders = int(np.ceil(np.log2(noValues)))
    noValuesGenerated = 2**noOrders
    
    y = np.array([1])
    for n in range(noOrders):
        y = next_step_1d(y, p)
    
    if (slope):
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = y
    
    return x[0:noValues], y[0:noValues]

     
def next_step_1d(y, p):
    y2 = np.zeros(y.size*2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*y.size:2] = y + sign*(1-2*p)*y
    y2[1:2*y.size+1:2] = y - sign*(1-2*p)*y
    
    return y2


def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
        
    a[1] = 0
    
    return a

#Endogenous (setup: N, p: 0.32-0.42, beta=0.4)
x, y = pmodel (256, 0.499, 0.4)
y=y-1
pyplot.plot(y)
pyplot.title("Endogenous Series")
plt.ylabel("Valores de Amplitude")
plt.xlabel("N passos no tempo")
pyplot.show()


#Exogenous (setup: N, p: 0.18-0.28, beta=0.7)
x, y = pmodel (256, 0.45, 0.7)
y=y-1
pyplot.plot(y)
pyplot.title("Exogenous Series")
plt.ylabel("Valores de Amplitude")
plt.xlabel("N passos no tempo")
pyplot.show()
