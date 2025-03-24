import matplotlib
import pandas as pd
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype
from seaborn import scatterplot
matplotlib.use('TkAgg')
from scipy.cluster.hierarchy import dendrogram


from vfv import *

def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v] = t[v].fillna(t[v].mean())
            else:
                t[v] = t[v].fillna(t[v].mode()[0])

def unique(a):
    k = np.unique(a, return_index=True)[1]
    return [a[i] for i in sorted(k)]


def partitie(h, k=None):
    m = np.shape(h)[0]
    n = m + 1
    if k is None:
        d = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(d)
        k = m - j
    else:
        j = m - k
    threshold = (h[j, 2] + h[j + 1, 2]) / 2
    c = np.arange(n)
    for i in range(m - k + 1):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k2] = n + i
        c[c == k1] = n + i
    coduri = pd.Categorical(c).codes
    p = np.array(["C" + str(i + 1) for i in coduri])
    return k, threshold, p

def plot_correlation_heatmap(data, numeric_columns):
    numeric_data = data[numeric_columns]

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Matricea de corelație")
    plt.show()

def histograms(data, numeric_columns):
    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=15, color='blue')
        plt.title(f'Histogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.show()

def boxplots(data, numeric_columns):
    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=data[column])
        plt.title(f'Boxplot for {column}')
        plt.ylabel(column)
        plt.show()


#Determinarea particiilor optime folosind funcția partitie
def partitie(h, k=None):
    m = np.shape(h)[0]
    n = m + 1
    if k is None:
        d = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(d)
        k = m - j
    else:
        j = m - k
    threshold = (h[j, 2] + h[j + 1, 2]) / 2
    c = np.arange(n)
    for i in range(m - k + 1):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k2] = n + i
        c[c == k1] = n + i
    coduri = pd.Categorical(c).codes
    p = np.array(["C" + str(i + 1) for i in coduri])
    return k, threshold, p

def plot_ierarhie(h, threshold, titlu, k, etichete):
    fig = plt.figure(titlu, figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    dendrogram(h, ax=ax, color_threshold=threshold,labels=etichete)
    plt.savefig("out/dendr_"+str(k))

def plot_indecsi_silhouette(x, partitia, titlu, k):
    fig = plt.figure(titlu, figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    plot_silhouette(x, partitia, titlu, ax=ax)
    plt.savefig("out/Silhouette_" + str(k))

def plot_scoruri(t, v1, v2, y, clase=None, titlu="Plot instante in axele principale",
                 etichete=False):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 14, "color": "b"})
    scatterplot(t, x=v1, y=v2, hue=y, hue_order=clase, ax=ax)
    if etichete:
        for j in range(len(t)):
            ax.text(t[v1].iloc[j], t[v2].iloc[j], t.index[j])
    plt.savefig("out/" + titlu + "_" + v1 + "_" + v2)

def histograme(t, variabila, partitie):
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle("Histograme pentru variabila " + variabila)
    assert isinstance(fig, plt.Figure)
    clase = np.unique(partitie)
    q = len(clase)
    min_max = (t[variabila].min(), t[variabila].max())
    # print(min_max)
    ax = fig.subplots(1, q, sharey=True)
    for i in range(q):
        axe = ax[i]
        assert isinstance(axe, plt.Axes)
        axe.set_xlabel(str(clase[i]))
        axe.hist(t[partitie == clase[i]][variabila], range=min_max, rwidth=0.9)
    plt.savefig("out/hist_"+variabila+"_"+str(q))

def plot_elbow(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='b')
    plt.title(title)
    plt.xlabel('Numărul de clusteri')
    plt.ylabel('Scor Elbow')
    plt.grid(True)
    plt.savefig("out/Elbow_Plot.png")
    plt.show()

def elbow_score(h, x):
    m = np.shape(h)[0]
    scores = []

    for k in range(1, m):
        if m - k < 0:
            break
        threshold = (h[m - k, 2] + h[m - k - 1, 2]) / 2
        c = np.arange(m + 1)
        for i in range(m - k + 1):
            k1 = h[i, 0]
            k2 = h[i, 1]
            c[c == k2] = m + i
            c[c == k1] = m + i
        coduri = pd.Categorical(c).codes
        p = np.array(["C" + str(i + 1) for i in coduri])
        silhouette = silhouette_score(x, p)
        scores.append(silhouette)

    return scores
