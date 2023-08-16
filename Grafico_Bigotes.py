import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


delimitador = ';'
archivo='Datos_Graficos.csv'
datos=pd.read_csv(archivo,delimitador)

fig, axes = plt.subplots(ncols=2)

for ax, (n,grp) in zip(axes, datos.groupby("Exp")):
    sns.boxplot(x="Exp", y="Calculado", data=grp, whis=np.inf, ax=ax)
    
    sns.swarmplot(x="Exp", y="Calculado", data=grp, 
                  palette=["r","r"], ax=ax,label="Calculado")
    
plt.legend()
fig1, axes = plt.subplots(ncols=2)

for ax, (n,grp) in zip(axes, datos.groupby("Exp")):
    sns.boxplot(x="Exp", y="Control", data=grp, whis=np.inf, ax=ax,color="g")
    
    sns.swarmplot(x="Exp", y="Control", data=grp, 
                  palette=["b","b"], ax=ax,label="Control")
    
    
plt.legend()
plt.show()
