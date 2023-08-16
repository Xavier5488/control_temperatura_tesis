import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

delimitador = ';'
archivo='Datos_Graficos.csv'
datos=pd.read_csv(archivo,delimitador)

datos.shape
datos.head()

colors = {1:"b",2:"b"}
colors1 = {1:"y",2:"y"}
datos_color= datos.Exp.map(colors)
datos_color1= datos.Exp.map(colors1)

fig,ax=plt.subplots()
ax.scatter(datos.Exp,datos.Control,color=datos_color,label="Control")
ax.scatter(datos.Exp,datos.Calculado,color=datos_color1,label="Calculado")

Experimento =  ['0','Normal', 'Calefactor']
mapeado = range(len(Experimento))
plt.xticks(mapeado, Experimento)
plt.xlabel("Experimento")
plt.ylabel("Temperatura en °C")
plt.legend()
plt.show()
