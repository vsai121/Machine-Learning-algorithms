import csv
import random
import math
import operator


# Split the data into training and test data
def loadDataset(filename, split, trainingSet=[] , testSet=[]):

    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])



def Distance(instance1 , instance2 , length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
                
    return math.sqrt(distance)


def getNeighbours(trainingSet , testInstance , k):

    distances = []
    length = len(testInstance)-1
    
    for i in range(len(trainingSet)):
        dist = Distance(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
        
    distances.sort(key = operator.itemgetter(1))
    
    neighbors = []
    
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors 
    
def getResponse(neighbours):
    classVotes = {}
    
    for i in range(len(neighbours)):
        response = neighbours[i][-1]
        
        if response in classVotes:
            classVotes[response] += 1
        
        else:
            classVotes[response] = 1
            
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0] 
    
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
    
        if testSet[x][-1] == predictions[x]:
            correct += 1
            
    return (correct/float(len(testSet))) * 100.0       
    


def main():
    trainingSet=[]
    testSet=[]
    split = 0.75
    loadDataset('iris.data', split, trainingSet, testSet)
    
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbours(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)

    accuracy = getAccuracy(testSet, predictions)
    print 'Accuracy: ', accuracy

main()       

