import numpy as np
import pandas as pd
import time
import sys
from future.utils import iteritems
from scipy.stats import multivariate_normal as mvn
from helpers import featureExctractor, constants, getData, util, statisticsWriter

class NaiveBayesClass:
    def __init__(self, n_iters=3):
        self.rawTrainingData = None
        self.trainingLabels = None
        self.rawValidationData = None
        self.validationLabels = None
        self.rawTestData = None
        self.testLabels = None
        self.legalLabels = None

        self.trainingData = None
        self.validationData = None
        self.testData = None
        
        self.n_iters = n_iters

        self.cond_prob_table = {'d':{0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}, 'f':{0:{},9:{}}}
        self.statistics = {}
    
    
    def getFeatures(self, dataType = 'd'):
        if dataType == 'd':
            self.trainingData = list(map(featureExctractor.Digit, self.rawTrainingData))
            self.testData = list(map(featureExctractor.Digit, self.rawTestData))
        elif dataType == 'f':
            self.trainingData = list(map(featureExctractor.Face, self.rawTrainingData))
            self.testData = list(map(featureExctractor.Face, self.rawTestData))
        
        return True
    
    def train(self, X, Y, dataType, smoothing=1e-2):
        self.label_stats = {}
        self.prior_table = {}
        labels = range(10) if dataType=='d' else range(2)
        for l in labels:
            label_filter = X[Y == l]
            self.label_stats[l] = {
                'mean': label_filter.mean(axis=0),
                'var': label_filter.var(axis=0) + smoothing,
            }
            self.prior_table[l] = float(len(Y[Y == l])) / len(Y)

    def score(self, X, Y):
        prediction = self.classify(X)
        return np.mean(prediction == Y)
    
    def log_probability(self, X, mean, var):
        exponent = -0.5 * ((X - mean) ** 2) / var
        prefactor = -0.5 * np.log(2 * np.pi * var)
        log_prob = np.sum(prefactor + exponent, axis=1)
        return log_prob


    def classify(self, X):
        rows, columns = X.shape
        num = len(self.label_stats)
        temp = np.zeros((rows, columns))
        label_max = -1
        for l, stats in iteritems(self.label_stats):
            mean, var = stats['mean'], stats['var']
            temp[:,l] = self.log_probability(X, mean, var) + np.log(self.prior_table[l])
            label_max = l
        return np.argmax(temp[:,:label_max+1], axis=1)

    def run(self, iters = 5, debug=False):

        self.testIters = iters

        if debug:
            TRAINDATA_SIZE = [0.1]
        else:
            TRAINDATA_SIZE = np.arange(0.1, 1.1, 0.1)
        
        dataTypes = ['d','f']# Digits and Faces.

        for dataType in dataTypes:

            self.statistics[dataType] = {}

            if dataType == 'f':
                dataSize = constants.FACE_TRAINING_DATA_SIZE
            else:
                dataSize = constants.DIGITS_TRAINING_DATA_SIZE

            for size in TRAINDATA_SIZE:
                acc = []
                avgTime = []

                for index in range(self.testIters):
                    trainingSize = int(size * dataSize)

                    # Fetch Data.
                    [self.rawTrainingData, 
                    self.trainingLabels,  
                    self.rawTestData, 
                    self.testLabels,
                    self.legalLabels] = getData.fetch(dataType, trainingSize)

                    # Convert raw data into features we want.
                    self.getFeatures(dataType=dataType)

                    trainData = pd.DataFrame(self.trainingData).to_numpy()
                    testData = pd.DataFrame(self.testData).to_numpy()
                    trainLabels = np.array(self.trainingLabels)
                    testLabels = np.array(self.testLabels)
                    
                    self.train(trainData,trainLabels,dataType)

                    print(f'Naive Bayes Training with {dataType} data and size {trainingSize}[{int(size*100)}%] and iteration {index}.....')
                    testStart = time.time()
                    test_accuracy = self.score(trainData,trainLabels)
                    testTime = time.time() -  testStart
                    avgTime.append(testTime)

                    print(f'Naive Bayes Testing Accuracy with {dataType} data and size {trainingSize}: {test_accuracy*100}%')

                    print(f'Naive Bayes Testing with {dataType} data and size {trainingSize}[{int(size*100)}%] and iteration {index}.....')
                    prediction_accuracy = self.score(testData,testLabels)
                    acc.append(prediction_accuracy)

                    print(f'Naive Bayes Prediction Accuracy with {dataType} data and size {trainingSize}[{int(size*100)}%]: {prediction_accuracy * 100}[iteration {index}]')
            
                # Once we have finished iterations on 10%, 20%, 30%....
                acc = np.array(acc)
                avgTime = np.array(avgTime)
                
                self.statistics[dataType][int(size*100)] = {}
                self.statistics[dataType][int(size*100)]['mean'] = np.mean(acc)
                self.statistics[dataType][int(size*100)]['std'] = np.std(acc)
                self.statistics[dataType][int(size*100)]['avgTime'] = np.mean(avgTime)
            
            print()
            
        return self.statistics
    
    def write(self):
        statisticsWriter.write(self.statistics, self.testIters, 'output/naivebayes_digit_results.txt', 'output/naivebayes_face_results.txt')

# Testing Process.
if __name__ == '__main__':
    classifierOne = NaiveBayesClass()
    classifierOne.run(debug=True)
    classifierOne.write()
