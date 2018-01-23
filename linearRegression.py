import numpy as np
import matplotlib.pyplot as plt

datafile = 'ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True) #Read in comma separated data

X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size 

X = np.insert(X,0,1,axis=1)#for bias term

#Gradient descent

iterations = 15000
alpha = 0.01

def hypothesis(theta , X):
    return np.dot(X , theta)
    


def computeCost(theta , X, y):
    #Cost function
    return float((1./(2*m)) * np.dot((hypothesis(theta,X)-y).T,(hypothesis(theta,X)-y))) 
 
"""    
initial_theta = np.zeros((X.shape[1],1)) #(theta is a vector with n rows and 1 columns (if X has n features) )
print computeCost(initial_theta,X,y)       
"""

#Gradient descent

def gradientDescent(X, theta_start = np.zeros(2)):
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on
    for _ in xrange(iterations):
        tmptheta = theta
        jvec.append(computeCost(theta,X,y))
        thetahistory.append(list(theta[:,0]))
        #Simultaneously updating theta values
        for j in xrange(len(tmptheta)):
            tmptheta[j] = theta[j] - (alpha/m)*np.sum((hypothesis(initial_theta,X) - y)*np.array(X[:,j]).reshape(m,1))
        theta = tmptheta
    return theta, thetahistory, jvec
    
    
initial_theta = np.zeros((X.shape[1],1))
theta, thetahistory, jvec = gradientDescent(X,initial_theta)

def plotConvergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*iterations,1.05*iterations])
    #dummy = plt.ylim([4,8])


plotConvergence(jvec)
dummy = plt.ylim([4,7])
plt.show()    


    

