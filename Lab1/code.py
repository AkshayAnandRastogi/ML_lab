import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
concepts = np.array(data)[:,:-1]
target = np.array(data)[:,-1]

def train(con,tar):
    for i,val in enumerate(tar):
        if val=='yes':
            specific_h = con[i].copy()
            break
            
    for i,val in enumerate(con):
        if tar[i]=='yes':
            for x in range (len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'
                else:
                    pass
    return specific_h

print(train(concepts,target))



Input:
sky	airtemp humidity	wind	water	forecast	enjoy sport
sunny	warm	normal	strong	warm	 same	    yes
sunny	warm	 high	  strong	warm	 same	    yes
rainy	cold	 high	  strong	warm	change	  no
sunny	warm	 high	  strong	cool	change	  yes


Output:
['sunny' 'warm' '?' 'strong' '?' '?']
