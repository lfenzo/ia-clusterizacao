import numpy as np

def dist(a, b) -> float:
    """
    Calcula a distância enclidiana entre dois vetores `a` e `b`
    """
    return np.sqrt( np.sum( (np.asarray(a) - np.asarray(b))**2 ) )
