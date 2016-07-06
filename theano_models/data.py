import numpy as np
from sklearn import cross_validation
import platform
import os

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)


def toy(random_state, n=20, noise=9):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    X = random_state.uniform(-4, 4, n)
    X.sort()
    Z = X ** 3 + random_state.normal(0, noise, n)

    Z = (Z - Z.mean(axis=0)) / Z.std(0)
    X = (X - X.mean(axis=0)) / (X.std(0) + 1e-4)

    X = X[:, np.newaxis].astype('float32')
    Z = Z[:, np.newaxis].astype('float32')

    return X, X, Z, Z


def _boston():
    D = np.loadtxt(os.path.join(__parent__,'data','bostonhousing','housing.data')).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return Z, X

def boston(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'bostonhousing', 'housing.data')).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def _concrete():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'concrete', 'concrete.csv'), delimiter=','
                   ).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return Z, X

def concrete(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data','concrete','concrete.csv'), delimiter=','
                   ).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)

def _energy():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'energy', 'energy.csv'), delimiter=','
                   ).astype('float32')
    X = D[:, :-2]
    Z = D[:, -2:-1]
    return Z, X

def energy(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'energy', 'energy.csv'), delimiter=','
                   ).astype('float32')
    X = D[:, :-2]
    Z = D[:, -2:-1]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def _kin8nm():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'kin8nm', 'regression-datasets-kin8nm.csv'),
                   delimiter=',').astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return Z, X

def kin8nm(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'kin8nm', 'regression-datasets-kin8nm.csv'),
                   delimiter=',').astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def _yacht():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'yacht', 'yacht_hydrodynamics.data')).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return Z, X

def yacht(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'yacht', 'yacht_hydrodynamics.data')).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)

def _naval():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'naval', 'data.txt')).astype('float32')
    X = D[:, :-2]
    Z = D[:, -2:-1]
    return Z,X

def naval(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'naval', 'data.txt')).astype('float32')
    X = D[:, :-2]
    Z = D[:, -2:-1]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / (X.std(0) + 1e-4)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)



def _powerplant():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'powerplant', 'powerplant.csv'), delimiter=',',
                   skiprows=1).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return X, Z

def powerplant(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'powerplant', 'powerplant.csv'), delimiter=',',
                   skiprows=1).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def _winered():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'winered', 'winequality-red.csv'), delimiter=';',
                   skiprows=1).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    return Z, X

def winered(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'winered', 'winequality-red.csv'), delimiter=';',
                   skiprows=1).astype('float32')
    X = D[:, :-1]
    Z = D[:, -1:]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / (X.std(0) + 1e-8)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def _protein():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'protein', 'CASP.csv'), delimiter=',',
                   skiprows=1).astype('float32')
    X = D[:, 1:]
    Z = D[:, :1]
    return X, Z

def protein(random_state):
    D = np.loadtxt(os.path.join(__parent__, 'data', 'protein', 'CASP.csv'), delimiter=',',
                   skiprows=1).astype('float32')
    X = D[:, 1:]
    Z = D[:, :1]
    Z -= Z.mean(axis=0)
    X = (X - X.mean(axis=0)) / X.std(0)
    return cross_validation.train_test_split(X, Z, test_size=.1,
                                             random_state=random_state)


def jura():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'jura', 'prediction.dat'), skiprows=13)
    X = D[:, (0, 1, 8, 10)]
    Z = D[:, 4:5]

    D = np.loadtxt(os.path.join(__parent__, 'data', 'jura', 'prediction.dat'), skiprows=13)
    TX = D[:, (0, 1, 8, 10)]
    TZ = D[:, 4:5]

    zm, zs = Z.mean(0), Z.std(0)
    xm, xs = X.mean(0), X.std(0)

    X = (X - xm) / xs
    TX = (TX - xm) / xs

    Z = (Z - zm) / zs
    TZ = (TZ - zm) / zs

    return X, TX, Z, TZ


def _year():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'year', 'YearPredictionMSD.txt'), delimiter=','
                   ).astype('float32')
    X = D[:, 1:]
    Z = D[:, :1]
    return X, Z

def year():
    D = np.loadtxt(os.path.join(__parent__, 'data', 'year', 'YearPredictionMSD.txt'), delimiter=','
                   ).astype('float32')
    X = D[:463715, 1:]
    Z = D[:463715, :1]
    TX = D[463715:, 1:]
    TZ = D[463715:, :1]

    xm, xs = X.mean(0), X.std(0)

    X = (X - xm) / xs
    TX = (TX - xm) / xs

    return X, TX, Z, TZ
