import matplotlib
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')

from functii import *
pd.set_option('display.max_columns', None)  # Afișează toate coloanele
pd.set_option('display.width', 1000)

set_date = pd.read_csv("DataIn/Data.csv", index_col=1)
variabile = list(set_date)[1:]

# Conversie coloane numerice la tipul de date numeric
numeric_columns = ["GDP(2022)", "Unemployment rate(2022)", "Inflation(2022)", "Exports(2022)", "Imports(2022)", "Foreign direct investments(2022)"]
set_date[numeric_columns] = set_date[numeric_columns].apply(pd.to_numeric, errors='coerce')

valori_lipsa = set_date.isna().any().any()
if valori_lipsa:
    nan_replace(set_date)

print("Statistici descriptive:\n", set_date.describe())

#Matrice de corelatie si heatmap
plot_correlation_heatmap(set_date, numeric_columns)

# Histograme pentru fiecare variabilă numerică
histograms(set_date, numeric_columns)

# Boxplot pentru identificarea valorilor extreme
boxplots(set_date, numeric_columns)

# Skewness și kurtosis
print("Skewness:\n", set_date[numeric_columns].skew())   # Asimetria distribuției
print("Kurtosis:\n", set_date[numeric_columns].kurtosis())  # Indicator al „grosimii” cozii distribuției

x = set_date[variabile].values

metoda = "ward"  #Metoda de legare a clusterelor
h = linkage(x, method=metoda)  # Construiește matricea de legături ierarhice
n = len(set_date)

print("Matricea ierarhie:")
# print(h)
print("Numar instante:",n)

print("Partitia optimala")

k_opt, threshold_opt, p_opt = partitie(h)

print("Numar clusteri:",k_opt)

# Calcul scor Silhouette la nivel de partitie -> cat de bine sunt separati in clustere
silhouette_opt = silhouette_score(x, p_opt)

# Calcul scor Silhouette la nivel de instante
index_silhouette_opt = silhouette_samples(x, p_opt)
print("Scor Silhouette:",silhouette_opt)
print("Partitia:")
for v in zip(set_date.index,p_opt,index_silhouette_opt):
    print(v)

plot_ierarhie(h, threshold_opt,
              "Plot ierarhie - partitia optimala. Scor Silhouette:" + str(silhouette_opt),
              k_opt, set_date.index)


# Creare tabel de partitii in care sunt salvate partitiile
t_partitii = pd.DataFrame({
    "P_opt": p_opt,
    "Scor Silhouette P_opt": index_silhouette_opt
}, index=set_date.index)

# Trasare grafic Silhouette
plot_indecsi_silhouette(x, p_opt, "Plot Silhouette. Metoda " + metoda, k_opt)


# Calcul axe principale pentru trasare plot partitie
pca = PCA(2)
pca.fit(x)
tz = pd.DataFrame(pca.transform(x), set_date.index, ["z1", "z2"])
#
# Plot partitie in axele principale
plot_scoruri(tz, "z1", "z2", p_opt, clase=np.unique(p_opt),
             titlu="Plot partitie optimala", etichete=False)

# Trasare histograme pentru partitia optimala
for variabila in variabile:
    histograme(set_date, variabila, p_opt)


# Calcul si analiza partitie cu k clusteri
# Exemplu pe partitia de 5 clusteri
k = 5
k, threshold_k, p_k = partitie(h, k=k)

silhouette_k = silhouette_score(x, p_k)
index_silhouette_p_k = silhouette_samples(x, p_k)

print("Partitia din ",k," clusteri")
print("Scor Silhouette:",silhouette_k)
print("Partitia:")
for v in zip(set_date.index,p_k,index_silhouette_p_k):
    print(v)

plot_ierarhie(h, threshold_k, "Plot ierarhie cu " + str(k) + " clusteri", k,
              set_date.index)
t_partitii["P_" + str(k)] = p_k
t_partitii["Scor Silhouette P_"+str(k)]=index_silhouette_p_k
plot_indecsi_silhouette(x, p_k, "Plot Silhouette. Metoda " + metoda, k)
plot_scoruri(tz, "z1", "z2", p_k, clase=np.unique(p_k),
             titlu="Plot partitie cu " + str(k) + " clusteri", etichete=False)

for variabila in variabile:
    histograme(set_date, variabila, p_k)

# Salvare partitii
t_partitii.to_csv("out/Partitii.csv")

# Calcul scor Elbow
elbow_scores = elbow_score(h, x)

# Plot Elbow
plot_elbow(range(1, len(elbow_scores) + 1), elbow_scores, "Elbow Plot")
