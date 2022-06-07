//Candidate Elimination Algorithm

import pandas as pd
import numpy as np
import re
df = pd.read_csv("C://Users//admin//Downloads//data.csv")
d = df.iloc[:,:-1]
print("The attributes are: ",d.columns.values)

target = df.iloc[:,-1]
print("The target is: ",target.values)
def generalizeS(d1,S):
    for i,j in enumerate(zip(d1,S[0])):
        if j[0] != j[1] and j[1] != '?':
            if j[1] != None:
                S[0][i] = '?'
            else:
                S[0][i] = j[0]
def specializeG(G,s1,d1):
    for i,j in enumerate(zip(G[0],s1,d1)):
        temp = G[0].copy()
        if j[0] != j[1] and j[0] == '?' and j[1] != j[2]:
            temp[i] = j[1]
            G.append(temp)
    G.pop(0)
def candidate_elimination(d,t):
    n = len(t)
    S = [[None for i in range(len(d[0]))]]
    G = [['?' for i in range(len(d[0]))]]
    neg = []
    pos = []
    for i in range(n):
        if t[i] == 'yes': pos.append(i)
        else: neg.append(i)

    # for positive instance
    for i in pos:
        generalizeS(d[i],S)
    # for positive instance
    for j in neg:
        specializeG(G,S[0],d[j])

    print("Specific Hypothesis: ", S[0], sep='\n')
    print("\nGeneral Hypothesis: ")
    for i in G:
        print(i)
    
candidate_elimination(d.values,target.values)
