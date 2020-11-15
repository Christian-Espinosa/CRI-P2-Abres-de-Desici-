import numpy as np
import time
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

    n = pd.factorize(dataset[dataset.keys()[-1]].values)
    dataset[dataset.keys()[-1]] = n[0]
    dataset = dataset.apply(pd.to_numeric, errors='coerce')

    dataset1 = dataset.dropna()
    dataset1 = dataset1.reset_index(drop=True)
    # eliminar columnas con valores no binarios(CAMBIAR POR FACTORIZE)
    # dataset1 = dataset1.drop(columns=[0, 1, 2])
    p = pd.factorize(dataset[dataset.keys()[0]])
    dataset[dataset.keys()[0]] = p[0]
    p = pd.factorize(dataset[dataset.keys()[1]])
    dataset[dataset.keys()[1]] = p[0]
    p = pd.factorize(dataset[dataset.keys()[2]])
    dataset[dataset.keys()[2]] = p[0]
    return dataset1

"""

def entropia(ds, atributo=None):
    # ID3
    eps = np.finfo(float).eps
    ent = 0
    target = list(ds.keys())[-1]
    t_v = ds[target].unique()

    if atributo is None:
        for v in t_v:
            aux = ds[target].value_counts()[v]/len(ds[target])
            ent += -aux*np.log2(aux)
    else:
        vs = ds[atributo].unique()
        for v in vs:
            ent_v = 0
            for t in t_v:
                n = len(ds[atributo][ds[atributo] == v][ds[target] == t])
                d = len(ds[atributo][ds[atributo] == v])
                f = n/(d+eps)
                ent_v += -f*np.log2(f+eps)
            f2 = d/len(ds)
            ent += f2*ent_v
    return abs(ent)
"""


def simple_entr(ds, atr=None):
    target = ds.keys()[-1]
    t_v = ds[target].unique()
    ent = 0
    if atr is None:
        for v in t_v:
            aux = ds[target].value_counts()[v]/len(ds[target])
            ent += -aux*np.log2(aux)
    else:
        pos_v = ds[atr].unique()
        for val in pos_v:
            # entropia parcial
            part_ent = 0
            # valores del atributo iguales a val
            atr_val = np.argwhere([ds[atr] == val])[:, 1]
            # numero de valores del atributo iguales a val
            atr_prop = len(atr_val)
            for f_val in t_v:
                s_vals = np.take(ds[target].values, atr_val)
                s_prop = len(s_vals[s_vals == f_val])
                p = (s_prop/atr_prop) if s_prop > 0 else 0
                part_ent += -p*np.log2(p) if p > 0 else 0
            ent += part_ent*(atr_prop/len(ds[atr]))
    return ent


def mejor_attr(ds, criterio):
    # Selección del mejor atributo segun el criterio de decisión
    if criterio == 0:
        gains = [gains_bruh(ds, atr) for atr in ds.keys()[:-1]]
        return ds.keys()[:-1][np.argmax(gains)]
    else:
        return


def gains_bruh(s, a):
    return simple_entr(s)-simple_entr(s, a)


def arbol_rec(ds, criterio, target, arbol=None):
    arbol = {}
    # Selección del mejor atributo
    atr = mejor_attr(ds, criterio)
    # Obtenemos los diferentes valores que puede tomar el atributo
    pos_vals = np.unique(ds[atr])
    # creamos nodo atributo
    arbol[atr] = {}
    # Construcción del arbol
    for val in pos_vals:
        indices = np.hstack(np.argwhere(ds[atr].values == val))
        new_ds = ds.drop(index=indices).reset_index(drop=True)
        # new_ds = ds[ds[atr] == val].reset_index(drop=True)
        new_ds = new_ds.drop(columns=[atr])
        valores, counts = np.unique(new_ds[target], return_counts=True)
        # Si hemos encontrado un solo valor para el atributo
        if len(counts) == 1:
            arbol[atr][val] = valores[0]
        # Si no llamamos recursivamente a la función
        else:
            arbol[atr][val] = arbol_rec(new_ds, criterio, target)
    return arbol


def predict(x_test, arbol):
    if arbol.dtype is not dict:
        return arbol
    else:
        return



def main():
    dataset = load_dataset('data/ad.data')
    dataset = treatdata(dataset)
    # particionamos el dataset con un 80% de training y el resto test(20%)
    train_set = dataset.drop(dataset.index[1887:])
    test_set = dataset.drop(dataset.index[:1887])
    # ds = dataset.drop(dataset.index[400:])
    criterio = 0# ID3
    target = dataset.keys()[-1]
    beg = time.time()
    # entrenamos a nuestro clasificador( creamos el arbol)
    arbol = arbol_rec(train_set, criterio, target)
    end = time.time()
    total = (end-beg)
    print(total)
    print(arbol)

if __name__ == '__main__':
    main()