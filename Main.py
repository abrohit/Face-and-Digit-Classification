from helpers import readData

from models.perceptron import PeceptronClass
from models.NaiveBayes import NaiveBayesClass
from models.knn import KNNClass

#Generate Statistics for each classifier.
if __name__ == '__main__':
    
    classifierOne = PeceptronClass()
    classifierOne.run()
    classifierOne.write()

    classifierTwo = NaiveBayesClass()
    classifierTwo.run()
    classifierTwo.write()

    classifierTwo = KNNClass()
    classifierTwo.run()
    classifierTwo.write()
