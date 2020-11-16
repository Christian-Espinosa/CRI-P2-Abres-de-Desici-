import numpy as np
import Node
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
    # dataset1 = dataset1.drop(columns=[4, 5, 6])
    s, b = pd.qcut(dataset1[dataset.keys()[0]], q=2, retbins=True, labels=False)
    dataset1[dataset.keys()[0]] = pd.cut(dataset1[dataset.keys()[0]], bins=b, labels=False)
    s, b = pd.qcut(dataset1[dataset.keys()[1]], q=2, retbins=True, labels=False)
    dataset1[dataset.keys()[1]] = pd.cut(dataset1[dataset.keys()[1]], bins=b, labels=False)
    s, b = pd.qcut(dataset1[dataset.keys()[2]], q=2, retbins=True, labels=False)
    dataset1[dataset.keys()[2]] = pd.cut(dataset1[dataset.keys()[2]], bins=b, labels=False)
    dataset1 = dataset1.reset_index(drop=True)
    dataset1.dropna()
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


def fast_entr(atr):
    values, value_count = np.unique(atr, return_counts=True)
    entr = np.sum([(-value_count[x]/np.sum(value_count))*np.log2(value_count[x]/np.sum(value_count))
                   for x in range(len(values))])
    return entr


def fast_gain(ds, atr, target):
    S = fast_entr(ds[target])
    values, value_count = np.unique(ds[target], return_counts=True)
    rel_entr = np.sum([(value_count[x]/np.sum(value_count))*fast_entr(ds.where(ds[atr] == values[x]).dropna()[target])
                       for x in range(len(values))])
    return S-rel_entr


def score(sco, x_tr, x_s):
    sco.fit(x_tr, x_tr.keys()[-1])
    return sco.score(x_s, x_s.keys()[-1])


def cross_validation(dataset, cv):
    criterio = 0  # ID3
    target = dataset.keys()[-1]
    f = np.array_split(dataset, cv)
    score = []
    for i in range(cv):
        x_tr = f.copy()
        x_s = f[i]
        x_tr = pd.concat(x_tr, sort=False)
        arbol = grow_tree(dataset, target, 0, 2)
        sco = accuracy(arbol, x_tr.copy(), x_s.copy())
        score.append(sco)
    return score


def simple_entr(ds, atr=None):
    target = ds.keys()[-1]
    t_v = ds[target].unique()
    ent = 0
    if atr is None:
        for v in t_v:
            aux = ds[target].value_counts()[v]/len(ds[target])
            ent += -aux*np.log2(aux)
    else:
        pos_v = np.unique(ds[atr])
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


def gini(ds, atr=None):
    target = ds.keys()[-1]
    g = 1
    if atr is None:
        values, counts = np.unique(ds[target], return_counts=True)
        for val in values:
            g -= (counts[val]/np.sum(counts))*(counts[val]/np.sum(counts))

def mejor_attr(ds, criterio):
    # Selección del mejor atributo segun el criterio de decisión
    target = ds[ds.keys()[-1]]
    if criterio == 0:
        gains = [gains_bruh(ds, atr) for atr in ds.keys()[:-1]]
        return ds.keys()[:-1][np.argmax(gains)]
    else:
        return


def gains_bruh(s, a):
    return simple_entr(s)-simple_entr(s, a)


def arbol_rec(ds, criterio, target,c , arbol=None):

    if c < 5:

        arbol = {}
        # Selección del mejor atributo
        atr = mejor_attr(ds, criterio)
        # Obtenemos los diferentes valores que puede tomar el atributo
        pos_vals = np.unique(ds[atr])
        # creamos nodo atributo
        arbol[atr] = {}
        # Construcción del arbol
        for val in pos_vals:
            # indices_v = np.hstack(np.argwhere(ds[atr].values == val))if len(np.argwhere(ds[atr].values == val)) > 0 else []
            # indices_nv = np.hstack(np.argwhere(ds[atr].values != val))if len(np.argwhere(ds[atr].values != val)) > 0 else []
            # new_ds = ds.drop(index=indices_nv).reset_index(drop=True)
            new_ds = ds[ds[atr] == val].reset_index(drop=True)  # AIXO ESTÀ BE PERO PROVO AMB DROP
            # new_ds = new_ds.drop(columns=[atr])
            valores, counts = np.unique(new_ds[target], return_counts=True)
            # new_ds = ds.drop(index=indices_nv).reset_index(drop=True)
            new_ds = new_ds.drop(columns=[atr])
            # Si hemos encontrado un solo valor para el atributo
            c += 1
            if len(counts) == 1:
                arbol[atr][val] = valores[0]
            # Si no llamamos recursivamente a la función
            else:
                arbol[atr][val] = arbol_rec(new_ds, criterio, target, c)

    return arbol


def grow_tree(ds, target, depth, max_depth=5, father=None):

    arbol = {}
    possible_values, count = np.unique(ds[target], return_counts=True)
    if len(possible_values) == 1:
        return possible_values[0]
    elif ds.shape[1] == 0:
        return possible_values[np.argmax(count)]
    elif len(ds) == 0 or depth >= max_depth:
        return father
    else:
        father = possible_values[np.argmax(count)]
        best_feat = mejor_attr(ds, 0)
        arbol[best_feat] = {}
        vals, cnt = np.unique(ds[best_feat], return_counts=True)
        for v in vals:
            depth += 1
            arbol[best_feat][v] = grow_tree(ds[ds[best_feat] == v].reset_index(drop=True), target, depth, max_depth, father)
        return arbol


def accuracy(y_target, y_pred):
    return len([x for x, x1 in zip(y_pred, y_target) if x == x1])/len(y_pred)


def predict(x, arbol, key, depth):

    for i in range(depth):
        arbol = arbol[key][x[key]]
        # a = a[x[0][key]]
        if type(arbol) != dict:
            return arbol
        key = list(arbol.keys())[0]

def main():

    dataset = load_dataset('data/ad.data')
    dataset = treatdata(dataset)
    # particionamos el dataset con un 80% de training y el resto test(20%)
    # new_targ = [1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    train_set = dataset.drop(dataset.index[1887:])
    test_set = dataset.drop(dataset.index[:1887])
    # ds = dataset.drop(dataset.index[400:])

    criterio = 0# ID3
    target = dataset.keys()[-1]
    # train_set[target] = new_targ
    beg = time.time()
    # entrenamos a nuestro clasificador( creamos el arbol)
    # arbol = arbol_rec(train_set, criterio, target, 0)
    # tree = grow_tree(train_set, target, 0, 2)
    tree_2 = {1243: {0: {351: {0: 1, 1: 1}}, 1: 1}}
    end = time.time()
    total = (end-beg)
    y_pred = []
    for x in test_set.values:
        y_pred.append(predict(x, tree_2, list(tree_2.keys())[0], 2))
    y_pred = np.hstack(y_pred)
    y_targ = test_set[test_set.keys()[-1]].values
    score = accuracy(y_targ, y_pred)
    print(total)
    print(tree_2)

if __name__ == '__main__':
    main()