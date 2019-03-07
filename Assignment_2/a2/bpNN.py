from starter import *
import numpy as np


'''
class bpNN:
    def __init__(self):
        self.trainData, self.validData, self.testData, self.trainTarget, self.validTarget, self.testTarget = loadData()
        self.newtrain, self.newvalid, self.newtest = convertOneHot(self.trainTarget, self.validTarget, self.testTarget)

        self.variance = 2/(28*28+10)
        self.mean, self.stand_dev = 0, math.sqrt(self.variance)
        
        self.Wo = np.random.normal(self.mean, self.stand_dev, 1000*10)
        self.bo = np.zeros(1*10)
        self.Wh = np.random.normal(self.mean, self.stand_dev, 10*1000)
        self.bh = np.zeros(1*1000)

        self.deltaW = np.full((10*1000),1e-5)
        self.deltaW = np.full((1000),1e-5)

        self.gamma = 0.99
        self.ephocs = 1000

    
        
if __name__ == '__main__':
    myNN = bpNN()
    print (myNN.variance)
'''