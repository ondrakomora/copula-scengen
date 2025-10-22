import numpy as np

from copula_scengen.modules.copula.empirical_copula import EmpiricalCopula

EmpiricalCopula(data=np.array([[0, 0], [1, 1]])).evaluate(np.array([0.5, 0.2]))
