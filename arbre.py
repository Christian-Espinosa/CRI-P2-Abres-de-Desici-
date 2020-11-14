import numpy as np

# funcions i paquets per a visualitzacio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# paquet per a manipulacio de dades
import pandas as pd

# funcions i paquets per a carregar datasets especifics (exemples simples)
from sklearn import datasets

# funcions per a partició de dades
from sklearn.model_selection import train_test_split
# funcions per a classificacio kNN
from sklearn.tree import DecisionTreeClassifier
# funcions per a metriques de rendiment
from sklearn.metrics import accuracy_score,  precision_score, roc_curve,roc_auc_score, auc
from sklearn.metrics import confusion_matrix,plot_confusion_matrix


#funcions per a la validació creuada
from sklearn.model_selection import cross_val_score

# Visualitzarem només 3 decimals per mostra, i definim el num de files i columnes a mostrar
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 50)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=None, delimiter=',')
    return dataset

def treatdata(dataset):
    dataset = dataset.apply(pd.to_numeric, errors='coerce')

    dataset1 = dataset.dropna()
    dataset1 = dataset1.reset_index(drop=True)
    return dataset1


def entropia(ds, atributo=None):
    # ID3
    eps = np.finfo(float).eps
    ent = 0
    val = ds.play.unique()

    if atributo is None:
        for v in val:
            aux = ds.play.value_counts()[v]/len(ds.play)
            ent += -aux*np.log2(aux)
    else:
        vs = ds[atributo].unique()
        for v in vs:
            ent_v = 0
            for t in val:
                n = len(ds[atributo][ds[atributo] == v][ds.play == t])
                d = len(ds[atributo][ds[atributo] == v])
                f = n/(d+eps)
                ent_v += -f*np.log2(f+eps)
            f2 = d/len(ds)
            ent += -f2*ent_v
    return ent


def mejor_attr(ds, criterio):
    llaves = list(ds.keys())[:-1]
    # Seleccionem el millor atribut segons el guany d'informació o el ratio de guany
    if criterio == 'ID3':
        gains = [gains_bruh(ds, atr) for atr in llaves]
        return llaves[np.argmax(gains)]
    else:
        return


def gains_bruh(s, a):
    return entropia(s)-entropia(s, a);


def arbol_rec(ds,criterio, arbol=None):
    # Selección del mejor atributo
    atr = mejor_attr(ds)
    # Obtenemos los diferentes valores que puede tomar el atributo
    pos_vals = np.unique(ds[atr])
    if arbol is None:
        arbol = {}
        arbol[atr]
    # Construcción del arbol
    for val in pos_vals:
        new_ds = ds[ds[atr] == val].reset_index(drop=True)
        # CHRISTIAN SIGUE POR AQUÍ

def main():
    dataset = load_dataset('data/ad.data')
    dataset = treatdata(dataset)
    print(11111)


if __name__ == '__main__':
    main()