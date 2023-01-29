# Modules de base
import numpy as np
import matplotlib.pyplot as plt
# Module relatif à Gurobi
from gurobipy import *
import json
from pathlib import Path
import pandas as pd
from UTA.UTA_functions import Built_model_instances
from UTA.UTA_functions import create_variables
from UTA.UTA_functions import add_constraints
from UTA.UTA_functions import add_objective_function
from UTA.UTA_functions import s_score
from UTA.UTA_functions import si_k

EPSILON = 0.001

def create_model(list_alternatives,partial_categories,EPSILON,L):
    m=Model("Preferences")
    
    instances= Built_model_instances(list_alternatives=list_alternatives,L=L)
    X, X1 =instances
    m , variables = create_variables(m=m,list_alternatives=list_alternatives,L=L)
    
    m=add_constraints(list_alternatives=list_alternatives,m=m,variables=variables,X=X,X1=X1,partial_categories=partial_categories,EPSILON=EPSILON,L=L)
    
    m=add_objective_function(m=m,variable=variables)
    
    return m

reference_universities = np.array([ [27.5, 30, 8, 83, 55],
                                    [32.5, 37.5, 45, 45, 91.5],
                                    [25, 32.5, 16, 90, 25],
                                    [30, 35, 4, 75, 85],
                                    [25, 32.5, 24, 100, 100],
                                    [39, 40, 8, 100, 15] ])

partial_categories=[[0,2],
                    [1,1],
                   [2,2],
                   [3,2],
                   [4,1],
                   [5,0]]

m=create_model(reference_universities,partial_categories,EPSILON,2)
m.optimize()

# Création de la liste, en fait on recrée la matrice sik avec les valeurs optimales

X, X1 = Built_model_instances(list_alternatives=reference_universities,L=2)
l=si_k(m,X,u=3)
lo=[32.5, 37, 45, 45, 91.5] # Attention à ne pas prendre des coefficients en dehors de l'intervalle définit par min(i) et max(i) 
print(s_score(lo,l,X,4))