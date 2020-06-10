Pessoal,

Boa tarde.
Com o objetivo de agilizar e uniformizar a geração de series temporais
caóticas, desenvolvi os dois programas abaixo que podem ser utilizados
para a geração das respectivas famílias estocásticas conforme
objetivo do exercício 5.1.   
Atentem para as faixas que devem ser utilizadas para
geração das séries temporais.

Logístico 1D:  
rho: de 3.81 até 4.00

Henon (logistico 2D): 
a: de 1.350 até 1.420
b: de 0.210 até 0.310

Fora dessas faixas as series não são do tipo "noise"
e, portanto, os atributos calculados para
caracterizar cada familia irão comprometer as 
analises de agrupamento com K-means.

POR FAVOR, CONSIDEREM AS FAIXAS ACIMA. 
AS FAIXAS QUE ESTÃO NA LISTA
NÃO ESTÃO PRECISAS E PODEM PREJUDICAR AS 
ANALISES DE AGRUPAMENTO.

=======================================================
#Gerador de Mapa Logístico Caótico 1D: Atrator e Série Temporal
#1D Chaotic Logistic Map Generator: Attractor and Time Series
#Reinaldo R. Rosa - LABAC-INPE
#Version 1.0 for CAP239-2020

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

#chaotic logistic map is f(x) = rho*x*(1-x)  with rho in (3.81,4.00)
def Logistic(rho,tau,x,y):

    return rho*x*(1.0-x), tau*x


# Map dependent parameters
rho = 3.88
tau = 1.1
N = 256

# Initial Condition
xtemp = 0.001
ytemp = 0.001
x = [xtemp]
y = [ytemp]


for i in range(1,N):
  xtemp, ytemp = Logistic(rho,tau,xtemp,ytemp)
  x.append( xtemp )
  y.append( ytemp )


# Plot the Attractor
plot(x,y, 'b,')
plt.title("Logistic Chaotic Attractor")
plt.ylabel("Valores de Amplitude: A(t)")
plt.xlabel("Valores de Amplitude: A(t+tau)")
show()

# Plot the time series
plt.plot(x)
plt.title("Logistic Chaotic Noise")
plt.ylabel("Valores de Amplitude: A(t)")
plt.xlabel("N passos no tempo")
plt.show()

================

#Gerador de Mapa Logístico Caótico 2D (Henon Map): Atrator e Série Temporal
#2D Chaotic Logistic Map Generator (Henon Map): Attractor and Time Series
#Reinaldo R. Rosa - LABAC-INPE
#Version 1.0 for CAP239-2020

import numpy as np
import pandas as pd
from numpy import sqrt
import matplotlib.pyplot as plt

#2D Henon logistic map is noise-like with "a" in (1.350,1.420) and "b" in (0.210,0.310)

def HenonMap(a,b,x,y):

    return y + 1.0 - a *x*x, b * x
 
# Map dependent parameters
a = 1.40
b = 0.210
N = 100

# Initial Condition
xtemp = 0.1
ytemp = 0.3
x = [xtemp]
y = [ytemp]


for i in range(0,N):
  xtemp, ytemp = HenonMap(a,b,xtemp,ytemp)
  x.append( xtemp )
  y.append( ytemp )

# Plot the time series
plot(x,y, 'b,')
plt.title("Henon Chaotic Attractor")
plt.ylabel("Valores de Amplitude: Y")
plt.xlabel("Valores de Amplitude: X")
show()

# Plot the time series
plt.plot(y)
plt.title("Henon Chaotic Noise")
plt.ylabel("Valores de Amplitude: Y")
plt.xlabel("N passos no tempo")
plt.show()


