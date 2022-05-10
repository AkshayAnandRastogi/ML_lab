Algorithm
ID3 (Examples, Target_attribute, Attribute)

< Examples are the training examples. Target attribute is the attribute whose value is to be
predicted by the tree. Attributes is a list of other attributes that may be tested by the learned
decision tree. Returns a decision tree that correctly classifies the given Examples.>

*  Create a Root node for the tree
*  If all Examples are positive, Return the single-node tree Root, with label = +
*  If all Examples are negative, Return the single-node tree Root, with label = -
*  If Attributes is empty, Return the single-node tree Root, with label = most common value of Target attribute in Examples
*  Otherwise Begin
    *  A <-- the attribute from Attributes that best* classifies Examples
    *  The decision attribute for Root <-- A
    *  For each possible value, vi, of A,
        *   Add a new tree branch below Root, corresponding to the test A = vi
        *   Let Examples: vi, be the subset of Examples that have value vi for A
        *   If Examples: vi, is empty
                * Then below this new branch add a leaf node with label = most common value of Target attribute in Examples
                * Else below this new branch add the subtree
                        ID3(Examples:vi, Target attribute, Attributes - {A})
*  End
*  Return Root


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
