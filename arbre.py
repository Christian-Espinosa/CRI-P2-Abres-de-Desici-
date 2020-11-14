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
        vars = ds[atributo].unique()
        for v in vars:
            ent_v = 0
            for t in val:
                n = len(ds[atributo][ds[atributo]==v][ds.play==t])
                d = len(ds[atributo][ds[atributo] == v])
                f = n/(d+eps)
    return ent


def main():
    dataset = load_dataset('/data/ad.data')
    dataset = treatdata(dataset)


if __name__ == '__main__':
    main()