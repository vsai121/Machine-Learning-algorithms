import numpy as np
from pprint import pprint


#sample training data
x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])

#To check working of np.unique()
print(np.unique(x2, return_counts=True))
print(np.unique(x1 , return_counts=True))

def split(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}
        
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
  
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res
 
print(entropy(x1))    

def mutual_information(y, x):

    res = entropy(y)

    # Partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # Calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        #p - probability
        #y[x==v] is the
        res -= p * entropy(y[x == v])

    return res
    

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = split(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

X = np.array([x1, x2]).T
pprint(recursive_split(X, y))
