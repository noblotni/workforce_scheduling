# Modules de base
import numpy as np
import matplotlib.pyplot as plt
# Module relatif à Gurobi
from gurobipy import *
import json
from pathlib import Path
import pandas as pd
from UTA_functions import Built_model_instances
from UTA_functions import create_variables
from UTA_functions import add_constraints
from UTA_functions import add_objective_function
from UTA_functions import s_score
from UTA_functions import si_k
from UTA_functions import create_variables_app
from UTA_functions import add_constraints_app
from UTA_functions import s_score_app
from UTA_functions import get_cl

# Data directory
DATA_PATH = Path("./Preorder_instance.csv")
DATA_PATH1 = Path("./medium_instance_pareto.csv")

I=pd.read_csv(DATA_PATH)
I1=I['Classe'].to_numpy()
I0=I.drop(['Classe'], axis=1).to_numpy()
I1=[[k,I1[k]] for k in range(len(I1))]
EPSILON = 0.001

############################################################################################
######################## MODELE AVEC COEFFICIENTS NON APPRIS ###############################
############################################################################################

def create_model(list_alternatives,partial_categories,EPSILON,L):
    m=Model("Preferences")
    
    instances= Built_model_instances(list_alternatives=list_alternatives,L=L)
    X, X1 =instances
    m , variables = create_variables(m=m,list_alternatives=list_alternatives,L=L)
    
    m=add_constraints(list_alternatives=list_alternatives,m=m,variables=variables,X=X,X1=X1,partial_categories=partial_categories,EPSILON=EPSILON,L=L)
    
    m=add_objective_function(m=m,variable=variables)
    
    return m



m=create_model(I0,I1,EPSILON,2)
m.optimize()

# Création de la liste, en fait on recrée la matrice sik avec les valeurs optimales

X, X1 = Built_model_instances(list_alternatives=I0,L=2)
l=si_k(m,X,u=3)
# Attention à ne pas prendre des coefficients en dehors de l'intervalle définit par min(i) et max(i) 
#for k in range(I0.shape[0]):
    #print(str(k)+str(s_score(I0[k],l,X,2)))
V=pd.read_csv(DATA_PATH1)

def identify_classes(instances,model,X): # On crée une fonction qui va attribuer des classes à nos instance.
    
    V1=instances.to_numpy()
    V1=V1[1:]
    V1=(V1.T[1:]).T

    l=si_k(model,X,u=3)
    g=np.zeros(len(V1))
    Y=np.c_[V1,g]
    Y=pd.DataFrame(Y,columns =['profit','projects_done','long_proj_duration','Classe'])
    c1=0
    c2=0
    c3=0
    for k in range(V1.shape[0]):
        if s_score(V1[k],l,X,2)==0 :
            Y['Classe'][k]='Solution non acceptable'
            c1+=1
        elif s_score(V1[k],l,X,2)==1 :
            Y['Classe'][k]='Solution neutre'
            c2+=1
        else :
            Y['Classe'][k]='Solution satisfaisante'
            c3+=1
    #print(c1/V1.shape[0],c2/V1.shape[0],c3/V1.shape[0])

    Y.to_csv('Classification_instances.csv', index=False)
    return

identify_classes(instances=V,model=m,X=X)

############################################################################################
######################## MODELE AVEC COEFFICIENTS APPRIS ###################################
############################################################################################



def create_model_app(list_alternatives,partial_categories,EPSILON,L):
    m=Model("Preferences")
    
    instances= Built_model_instances(list_alternatives=list_alternatives,L=L)
    X, X1 =instances
    m , variables = create_variables_app(m=m,list_alternatives=list_alternatives,L=L)
    
    m=add_constraints_app(list_alternatives=list_alternatives,m=m,variables=variables,X=X,X1=X1,partial_categories=partial_categories,EPSILON=EPSILON,L=L)
    
    m=add_objective_function(m=m,variable=variables)
    
    return m

m_app=create_model_app(I0,I1,EPSILON,2)
m_app.optimize()

l=si_k(m_app,X,u=3)
cl=get_cl(m_app,X)

def identify_classes_appris(instances,model,X,cl): # On crée une fonction qui va attribuer des classes à nos instance.
    
    V1=instances.to_numpy()
    V1=V1[1:]
    V1=(V1.T[1:]).T

    l=si_k(model,X,u=3)
    g=np.zeros(len(V1))
    Y=np.c_[V1,g]
    Y=pd.DataFrame(Y,columns =['profit','projects_done','long_proj_duration','Classe'])
    print(cl)
    c1=0
    c2=0
    c3=0
    for k in range(V1.shape[0]):
        if s_score_app(V1[k],l,X,2,cl)==0 :
            Y['Classe'][k]='Solution non acceptable'
            c1+=1
        elif s_score_app(V1[k],l,X,2,cl)==1 :
            Y['Classe'][k]='Solution neutre'
            c2+=1
        else :
            Y['Classe'][k]='Solution satisfaisante'
            c3+=1
    #print(c1/V1.shape[0],c2/V1.shape[0],c3/V1.shape[0])

    Y.to_csv('Classification_instances_coeff_appris.csv', index=False)
    return

identify_classes_appris(instances=V,model=m,X=X,cl=cl)