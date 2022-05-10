import pandas as pd
import numpy as np
import math

df = pd.read_csv("C:\\Users\\admin\\Downloads\\ID3.csv")
df = df.iloc[:,1:6]

feature_names = df.columns[0:4].values
target = df.columns[-1]
print('Features in the dataset: ', feature_names)
print('Target in the dataset: ', target)

class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

def entropy(data):
    pos = neg = 0
    if len(data[target].unique()) == 2:
        pos, neg = data[target].value_counts()
    if pos == 0 or neg == 0:
        return 0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))

def total_gain(data, attr):
    uniq = data[attr].unique()
    gain = entropy(data)
    for u in uniq:
        subdata = data[data[attr] == u]
        gain -= (float(len(subdata)) / float(len(data))) * entropy(subdata)
    return gain

def ID3(data, features):
    root = Node()
    max_gain = 0
    choo_feat = ""
    for feature in features:
        gain = total_gain(data, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = df[max_feat].unique()

    for u in uniq:
        subdata = data[data[max_feat] == u]
        newNode = Node()
        if entropy(subdata) == 0:
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = subdata[target].unique()
        else:
            newNode.value = u
            new_attrs = features.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            newNode.children.append(child)
        root.children.append(newNode)
    return root

def printTree(root: Node, depth=0):
    print("\t"*depth,"|____", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" ---> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth + 1)
root = ID3(df, feature_names.tolist())
printTree(root)
