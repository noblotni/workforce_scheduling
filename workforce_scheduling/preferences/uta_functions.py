import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

def create_variables(m,list_alternatives,L):
    #Attention Potentially add the model as input unlike the other part.
    # Epsilon variables to deal with absolute values
    eps_plus = np.array([m.addVar(name=
                            "eps_plus_"
                            + str(j),
                            lb=0,
                        )
                        for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )
    eps_minus = np.array([
                            m.addVar(name=
                            "eps_minus_"
                            + str(j),
                            lb=0,)
                            for j in range(list_alternatives.shape[0])
                            ],
                        dtype=object,
    )
    s = np.array(
                [
                    [
                       m.addVar(name=
                            "s_"
                            + str(i)
                            + ","
                            + str(k),
                            lb=0,
                        )
                for k in range(L+1)
            ]
            for i in range(list_alternatives.shape[1])
        ],
        dtype=object,
    )
    variables = {"s": s, "eps_plus": eps_plus, "eps_minus": eps_minus}
    return m,variables

def add_constraints(list_alternatives,m,variables,X,X1,partial_categories,EPSILON,L): # For s(j) potentially generated a new matrix with the coefficient (xij-xik)/(xik+1-xik) which is actually determined at the beginning of the algorithm
    s=variables["s"]
    for p in partial_categories :
        o=quicksum([s[l][X1[p[0]][l]] + ((list_alternatives[p[0]][l]-X[l][X1[p[0]][l]])/(X[l][X1[p[0]][l]+1]-X[l][X1[p[0]][l]]))*(s[l][X1[p[0]][l]+1]-s[l][X1[p[0]][l]]) for l in range(list_alternatives.shape[1])])
        if p[1]==0:
            m.addConstr(o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]] <= 0.33, name= "c_"+str(p[0])+str(0))
        if p[1]==1:
            m.addConstr(0.33 + EPSILON <= o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]],name= "c_"+str(p[0])+str(1))
            m.addConstr(o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]] <= 0.66,name= "c_"+str(p[0])+str(3))
        if p[1]==2:
            m.addConstr(0.66 + EPSILON <= o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]],name= "c_"+str(p[0])+str(2))
    for i in range(list_alternatives.shape[1]):
        m.addConstr(s[i][0]==0, name="intervalle"+str(i))
        
    for i in range(list_alternatives.shape[1]):
        for k in range(L) :
            m.addConstr(s[i][k+1]-s[i][k]>=EPSILON, name="coeffs" + str(i)+str(k)) #esp can be different from EPSILON but we start by setting it equal to EPSILON
    
    m.addConstr(quicksum(s[i][L] for i in range(list_alternatives.shape[1]))==1,name="normalisation")
    
    return m

def Built_model_instances(list_alternatives,L) : #Cf calculation of s(j) / For this we define xik (the bounds of the intervals) and xijk (the k corresponding to xij: 3 dimensions.)
    
    X=np.zeros((list_alternatives.shape[1],L+1))
    O=list_alternatives.T
    for i in range(list_alternatives.shape[1]): # i this column of list_alternatives becomes a row for X
        for k in range(L+1):
            X[i][k]= min(O[i]) + (k/L)*(max(O[i])-min(O[i]))
    X1=np.ones(list_alternatives.shape,dtype = int)
    for i in range(list_alternatives.shape[0]):
        for j in range(list_alternatives.shape[1]): # For X1, X1 corresponds to the value of k for the coefficient ij, j cirtère, i instance
            for k in range(L):
                if X[j][k]<=list_alternatives[i][j]<= X[j][k+1] :
                    X1[i][j]=k
    return X,X1

def add_objective_function(m,variable):
    suma=quicksum([variable["eps_plus"][i] + variable["eps_minus"][i] for i in range(len(variable["eps_plus"]))])
    m.setObjective(suma , GRB.MINIMIZE)
    return m

def s_score(p,s,X,L):
    x1=[]
    for j in range(len(p)): # For X1, X1 corresponds to the value of k for the coefficient ij, j cirtère, i instance
        for k in range(L):
            if X[j][k]<=p[j]<= X[j][k+1] :
                x1.append(int(k))
    sm=0
    for l in range(len(p)):
        sm+=s[l][x1[l]] + ((p[l]-X[l][x1[l]])/(X[l][x1[l]+1]-X[l][x1[l]]))*(s[l][x1[l]+1]-s[l][x1[l]])
    if sm<=0.33 :
        return 0
    elif 0.33<sm<=0.66 :
        return 1
    else :
        return 2

def si_k(m,X,u):
    l=[]
    c=0
    i=[]
    for v in m.getVars():
        if v.VarName[0]=="s":
            c+=1
            i.append(v.X)
            if c%u==0 :
                l.append(i)
                i=[]
    return np.array(l)

def add_constraints_app(list_alternatives,m,variables,X,X1,partial_categories,EPSILON,L):
    s=variables["s"]
    for p in partial_categories :
        o=quicksum([s[l][X1[p[0]][l]] + ((list_alternatives[p[0]][l]-X[l][X1[p[0]][l]])/(X[l][X1[p[0]][l]+1]-X[l][X1[p[0]][l]]))*(s[l][X1[p[0]][l]+1]-s[l][X1[p[0]][l]]) for l in range(list_alternatives.shape[1])])
        if p[1]==0:
            m.addConstr(o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]] <= variables["cl"][0], name= "c_"+str(p[0])+str(0)) # Introduction of coefficients "cl" as variables because we want to learn them. 
        if p[1]==1:
            m.addConstr(variables["cl"][0] + EPSILON <= o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]],name= "c_"+str(p[0])+str(1))
            m.addConstr(o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]] <= variables["cl"][1],name= "c_"+str(p[0])+str(3))
        if p[1]==2:
            m.addConstr(variables["cl"][1] + EPSILON <= o-variables["eps_plus"][p[0]]+ variables["eps_minus"][p[0]],name= "c_"+str(p[0])+str(2))
    m.addConstr(0.15<=variables["cl"][0],name= "cl_1")
    m.addConstr(variables["cl"][0]+EPSILON <=variables["cl"][1],name="cl_2")
    for i in range(list_alternatives.shape[1]):
        m.addConstr(s[i][0]==0, name="intervalle"+str(i))
        
    for i in range(list_alternatives.shape[1]):
        for k in range(L) :
            m.addConstr(s[i][k+1]-s[i][k]>=EPSILON, name="coeffs" + str(i)+str(k))
    
    m.addConstr(quicksum(s[i][L] for i in range(list_alternatives.shape[1]))==1,name="normalisation")
    
    return m

def create_variables_app(m,list_alternatives,L): # Fonction for the model with coefficients learnt
    eps_plus = np.array([m.addVar(name=
                            "eps_plus_"
                            + str(j),
                            lb=0,
                        )
                        for j in range(list_alternatives.shape[0])
        ],
        dtype=object,
    )
    eps_minus = np.array([
                            m.addVar(name=
                            "eps_minus_"
                            + str(j),
                            lb=0,)
                            for j in range(list_alternatives.shape[0])
                            ],
                        dtype=object,
    )
   
    s = np.array(
                [
                    [
                       m.addVar(name=
                            "s_"
                            + str(i)
                            + ","
                            + str(k),
                            lb=0,
                        )
                for k in range(L+1)
            ]
            for i in range(list_alternatives.shape[1])
        ],
        dtype=object,
    )
    cl= np.array(
                [
                       m.addVar(name=
                            "cl_"
                            + str(k),
                            lb=0,
                        )
                for k in range(2)
            ],
        dtype=object,
    )
    variables = {"s": s, "eps_plus": eps_plus, "eps_minus": eps_minus,"cl":cl}
    return m,variables

def get_cl(m,X): # Get the learnt coefficients after solving the problem with Gurobi
    cl=[]
    for v in m.getVars():
        if v.VarName[0]=="c":
            cl.append(v.X)
    cl=np.array(cl)
    return cl

def s_score_app(p,s,X,L,cl):
    x1=[]
    for j in range(len(p)):
        for k in range(L):
            if X[j][k]<=p[j]<= X[j][k+1] :
                x1.append(int(k))
    sm=0
    for l in range(len(p)):
        sm+=s[l][x1[l]] + ((p[l]-X[l][x1[l]])/(X[l][x1[l]+1]-X[l][x1[l]]))*(s[l][x1[l]+1]-s[l][x1[l]])
    if sm<=cl[0] :
        return 0
    elif cl[0]<sm<=cl[1] :
        return 1
    else :
        return 2