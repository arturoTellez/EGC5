### Dependencias
# python-ternary
# plotly
# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
from scipy.stats import pareto, kstest

import ternary
import matplotlib.pyplot as plt

# Leer el archivo CSV
D = pd.read_csv(
    "expresion_genes_GS.csv",
    names = ["name", "col1", "col2", "col3"],
    sep = "\t"
)

print(D.head())

#Cada gen tiene un valor, no hay genes duplicados
print(D.name.value_counts())

"""1. Filtrar el archivo D, descartando a los genes cuya suma de las columnas 2 a 4 no sea mayor a 1
(uno). Llamaremos al archivo filtrado D’."""
D_prima = D[D.iloc[:, 1:].sum(axis = 1) > 1].copy()


"""
2. Convertir los genes en D’ a un esquema composicional. Los datos composicionales son aquellos en
los que la suma de los componentes es siempre la misma. Por ejemplo, sea G un gene (renglón) en D’.
Las columnas 2, 3 y 4 de G se denotan como G0, G1 y G2. El total de la expresión del gene G se obtiene
al sumar las tres cantidades: Total = G0+ G1 + G2. Ahora, el proceso de composionalidad se refiere a
dividir el valor de cada componente (Gi) entre el total: CG0 = G0/Total, CG1 = G1/Total y CG2 =
G2/Total, donde C Gi es el valor composicional de Gi G. Este paso se realiza para cada gene en D’, por
separado, con un Total independiente para cada gene. Esto equivale a añadir cuatro columnas a D’, las
tres primeras de ellas son los valores composicionales de expresión, y la cuarta columna es el total de
expresión."""


total_por_fila = D_prima.iloc[:, 1:].sum(axis = 1)
for i in range(1,4):
    D_prima.loc[:, "G_" + str(i)] =( D_prima.iloc[:, i] / total_por_fila).values


D_prima.loc[:, "G_Total"] = total_por_fila.values


"""3. Obtener el histograma de expresión total, sobre los genes en D’. Considerar 20 intervalos
homogéneos."""
D_prima.G_Total.hist(bins = 20)
plt.savefig('histograma.png')



"""4. El histograma obtenido en el paso anterior ¿Parece aproximar alguna función de probabilidad?
Explique.
El histograma se asemeja a una distribución de Pareto. Al realizar el ajuste de parámetros y aplicar una prueba de Kolmogorov-Smirnov (KS), observamos que no podemos rechazar la hipótesis de que los datos provienen de una distribución de Pareto. Los parámetros ajustados son:

Forma: 0.05115708985292857,
Localización: 1,
Escala: 1.0308400000269557e-07.
Con un valor p de 0.5, concluimos que no hay evidencia suficiente para rechazar la hipótesis de que los datos siguen una distribución de Pareto."""



# Ajustamos los parámetros de la distribución de Pareto a los datos filtrados
# Esta vez fijamos el parámetro 'loc' en 1, ya que la distribución de Pareto no debe tener un 'loc' negativo
pareto_params_filtered = pareto.fit(D_prima.G_Total, floc=1)

# Realizamos la prueba de Kolmogorov-Smirnov (KS) para la distribución de Pareto con los parámetros ajustados
ks_statistic, p_value = kstest(D_prima.G_Total, 'pareto', args=pareto_params_filtered)

# Imprimimos los parámetros y resultados de la prueba KS
print(pareto_params_filtered, ks_statistic, p_value)


print(D_prima.G_Total.describe(percentiles=np.linspace(0,1,21)))


"""5. Obtener la entropía de Shannon de la expresión total."""



def calcular_entropia_shannon(columna):
    # Contar la frecuencia de cada valor único en la columna
    frecuencias = columna.value_counts()

    # Calcular la probabilidad de cada valor único
    probabilidades = frecuencias / len(columna)

    # Aplicar la fórmula de la entropía de Shannon
    entropia = -np.sum(probabilidades * np.log2(probabilidades))

    return entropia


print("Entropia de Shannon: ", calcular_entropia_shannon(D_prima.G_Total))



"""6. Indagar técnicas de visualización de datos composicionales y hacer uso de alguna de ellas para
visualizar en dos dimensiones al conjunto de genes D’, haciendo uso de las variables (CG0, CG01 y
CG2 ). Se recomienda consultar el libro de Compositional Data Analysis in Practice, de Greenacre."""




df_normalized = D_prima[ ['G_1', 'G_2', 'G_3']]

df = px.data.election()
fig = px.scatter_ternary(df_normalized, a="G_1", b="G_2", c="G_3")
fig.write_image("ternary.png")


"""7. Aplicar los algoritmos de K-medias y DBSCCAN sobre las expresiones composicionales. Esto es,
los datos de entrada a estos dos algoritmos contarán con tres columnas: CG0, CG01 y CG2 . Sobre la
visualización obtenida en 5, indicar, por algún código de color, la etiqueta o clase obtenida por K-
medias y DBSCAN. Para k-medias, llevar este proceso para k = 1, 2, 3, 5, y 10. Para DBSCAN, utilizar
los parámetros que juzgue convenientes."""


dbscan = DBSCAN(eps=0.01, min_samples = 5, )
mapeo = dbscan.fit_predict(D_prima[ ['G_1', 'G_2', 'G_3']])

fig = px.scatter_ternary(D_prima, a="G_1", b="G_2", c="G_3", hover_name="name",
color=mapeo.astype(str),

)
fig.write_image("ternary_dbscan.png")


print(D_prima[mapeo == pd.DataFrame(mapeo).value_counts().idxmin()[0]])



for k in [1, 2, 3, 5, 10]:
    kmean = KMeans(n_clusters  = k)
    mapeo = kmean.fit_predict(D_prima[ ['G_1', 'G_2', 'G_3']])
    fig = px.scatter_ternary(D_prima, a="G_1", b="G_2", c="G_3", hover_name="name",
    color=mapeo.astype(str),
    size="G_Total", size_max=35,
    )
    fig.write_image(f"ternary_kmean{k}.png")

print(D_prima[mapeo == pd.DataFrame(mapeo).value_counts().idxmin()[0]])



"""En el análisis con K-means, no se identificaron grupos claramente definidos debido a la proximidad de los puntos entre sí, lo que impide la formación de grupos más pequeños y distintos. Al ajustar DBSCAN con un epsilon de 0.01 y un mínimo de 5 puntos por cluster (n=5), se detectaron agrupaciones de puntos que no fueron asignados a ningún cluster en el análisis anterior. En el caso de K-means, el cluster con menos puntos muestra una tendencia hacia valores cercanos a 1 para la columna G1. Por otro lado, en el análisis de DBSCAN, el cluster con menos puntos muestra valores de G1 centrados, lo que indica un comportamiento diferente al observado con K-means."""