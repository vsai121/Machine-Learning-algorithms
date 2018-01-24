import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import optimize

datafile = 'ex2data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data

X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size 

#bias term
X = np.insert(X,0,1,axis=1)

iterations = 5000
alpha = 0.001
    
"""
print(X.shape)
print(y.shape)

print(X[:,1])
"""

#For visualising data
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])

"""
print(pos.shape)
print(neg.shape)
"""

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)
    plt.show()
    
 
def sigmoid(x):
    print(x.shape)
    return 1 / (1 + np.exp(-x))   
    
def hypothesis(theta , X):
    Z = np.dot(X , theta)
    return sigmoid(Z)
    

def computeCost(theta , X , y , lamda=0):
    
    term1 = np.dot(y.T , np.log(hypothesis(theta , X)))
    term2 = np.dot((1-y).T , np.log(1-hypothesis(theta , X)))
    
    regterm = (lamda/2) * np.sum(np.dot(theta[1:].T,theta[1:]))   
    
    cost = float( (-1./m) * ( np.sum(term1 + term2) ) + float(lamda/(2*m))*regterm ) )

    return cost
    
"""
cost = computeCost(initial_theta,X,y)
    
print(cost)



def gradientDescent(X , theta_start = np.zeros((X.shape[1]))):
    theta = theta_start
    
    jvec=[]
    
    thetaHistory = []
    
    for i in xrange(iterations):
        
        tempTheta = theta
        cost = computeCost(theta , X , y)
        jvec.append(cost) 
        print(cost)
        for j in xrange(tempTheta.shape[0]):
            tempTheta[j] = theta[j] - (alpha/m)*np.sum((hypothesis(theta,X) - y)*np.array(X[:,j]).reshape(m,1))
  
        theta = tempTheta

    return theta , jvec
    
initial_theta = np.zeros((X.shape[1],1))
theta, jvec = gradientDescent(X,initial_theta)

def plotConvergence(jvec):
    plt.plot(range(len(jvec)),jvec)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.show()

plotConvergence(jvec)
plt.show() 

"""

def optimizeTheta(theta,X,y,lamda=0.):
    result = optimize.fmin(computeCost, x0=theta, args=(X, y, lamda), maxiter=400, full_output=True)
    return result[0], result[1]


initial_theta = np.zeros((X.shape[1],1))
theta, mincost = optimizeTheta(initial_theta,X,y)

print hypothesis(theta,np.array([1, 45.,85.]))


def makePrediction(theta, X):
    return hypothesis(theta,X) >= 0.5


    
    
