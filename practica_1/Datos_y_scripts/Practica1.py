import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import Clases_inst

rm = pyvisa.ResourceManager()

#gen_fun = rm.list_resources()[0]
#gen = rm.open_resource(gen_fun)
#
#osci = rm.list_resources()[1]
#os = rm.open_resource(osci)
##print(inst.query('*IDN?'))
#
#gen.write('FREQ 2000')
#gen.query_ascii_values('FREQ?')
#
#xze, xin, yze, ymu, yoff = os.query_ascii_values('WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;', separator=';') 
#data = (os.query_binary_values('CURV?', datatype='B',container=np.array) - yoff) * ymu + yze        
#tiempo = xze + np.arange(len(data)) * xin
#
#plt.plot(tiempo, data)
#plt.show()
#%%
import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import Clases_inst
import time

rm = pyvisa.ResourceManager()
gen_url, osci_url, _ = rm.list_resources()

osci = Clases_inst.TDS1002B(osci_url)
gen = Clases_inst.AFG3021B(gen_url)

#seteamos el generador
gen.setFrequency(10)
gen.getFrequency()

gen.setAmplitude(1)
gen.getAmplitude()

#seteamos el oscilscopio canal 1
osci.get_channel(1)
osci.set_channel(1, scale = 0.2, zero = 0)
osci.get_channel(1)

osci.set_time(0.5/10, zero  = 0)
osci.get_time()
t, d = osci.read_data(1)
plt.figure()
plt.plot(t,d, label  ='canal 1')
plt.legend()
plt.show()

#canal 2

osci.set_channel(2, scale = 0.2, zero = 0)
osci.set_time(0.5/10, zero  = 0)
osci.get_time()
t, d = osci.read_data(2)
plt.figure()
plt.plot(t,d, label = 'canal2')
plt.legend()
plt.show()
osci.get_channel(1)
osci.get_channel(2)

#%% Medir

import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import Clases_inst
import time

rm = pyvisa.ResourceManager()
gen_url, osci_url, _ = rm.list_resources()

osci = Clases_inst.TDS1002B(osci_url)
gen = Clases_inst.AFG3021B(gen_url)

#seteamos el generador
gen.setFrequency(10)
gen.setAmplitude(1) #que en realidad es 2V pico pico

#seteo inicial de ambos canales del osciloscopio
osci.set_channel(1, scale = 0.2, zero = 0)
osci.set_channel(2, scale = 0.2, zero = 0)
osci.set_time(0.02, zero  = 0)

t0 = time.time()
frecs =np.logspace(np.log10(10), np.log10(1000), 100)
salida_max = np.zeros(len(frecs))
salida_range = np.zeros(len(frecs))
for i,f in enumerate(frecs):
    gen.setFrequency(f)
    osci.set_channel(1, scale = 0.2, zero = 0)
    osci.set_channel(2, scale = 2/f, zero = 0)
    osci.set_time(0.5/f, zero  = 0)
    t, d = osci.read_data(2)
    salida_max[i] += max(d)
    salida_range[i] += osci.get_range(2)[1]

tf = time.time()

plt.figure()    
plt.plot(frecs, salida_max, '-o',label = 'max')
plt.plot(frecs, salida_range,'o' ,label  = 'range')
plt.legend()
plt.show()

print(tf-t0)

#%%
np.savetxt('datos_max.txt', salida_max)
np.savetxt('frecs.txt', frecs)
#%%
from scipy.optimize import curve_fit

def ajuste(x, wo):
    x = 2*np.pi*x
    return 1/np.sqrt(1+(x/wo)**2)


popt, pcov =  curve_fit(ajuste, frecs, salida_max, p0 = 175)

x_varios = np.linspace(np.log10(10), 1000,1000)

plt.plot(frecs, salida_max, '-o',label = 'max')
plt.plot(x_varios, ajuste(x_varios, popt), 'r-')