import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open('prueba.csv').readlines()[:-1], skiprows = 33, delimiter = ';')

freqs = data[:,0]
intensidad = data[:,1]

print(data.shape)	
print(freqs, intensidad)
plt.plot(freqs, intensidad)
plt.show()
