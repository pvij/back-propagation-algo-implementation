import time
import numpy as np
import matplotlib.pyplot as plt

class stochasticGradient :
    def __init__( self , kwargs ) :
        self.inputVectors               = kwargs["inputVectors"]
        self.expectedOutput             = kwargs["expectedOutput"]
        self.noOfEpochs                 = kwargs["noOfEpochs"]
        self.activationFnsForAllLayers  = kwargs["activationFnsForAllLayers"]
        self.noOfUnitsInEachLayer       = kwargs["noOfUnitsInEachLayer"]
        self.loss                       = kwargs["lossFn"]
        self.learningRate               = kwargs["learningRate"]
        self.batchSize                  = kwargs["batchSize"]
        self.noOfHiddenLayers           = len(self.noOfUnitsInEachLayer) - 2

    def start(self) :
        self.setInitialWeights()
        self.startAlgo()
        self.plotLoss()
        self.plotDecisionBoundary() #Can only be used in case of 2-D data

    def plotDecisionBoundary(self) :
        x_min = np.floor(min( self.inputVectors[:,0] ))
        x_max = np.ceil(max( self.inputVectors[:,0] ))
        y_min = np.floor(min( self.inputVectors[:,1] ))
        y_max = np.ceil(max( self.inputVectors[:,1] ))
        input   = [(x, y) for x in np.arange(x_min, x_max, .05) for y in np.arange(y_min, y_max, .05)]
        inputT  = np.array( input )
        output  = self.forwardPass( inputT )
        for i in range(len(output)):
            if output[i] == 0:
                plt.plot(input[i][0], input[i][1], 'co')
            elif output[i] < 0:
                plt.plot(input[i][0], input[i][1], 'r.')
            elif output[i] > 0:
                plt.plot(input[i][0], input[i][1], 'b.')
        self.plotData()
        plt.show()

    def plotData(self) :
        expectedOutputAsList    = list(self.expectedOutput[:])
        positiveIndices         = [i for i, x in enumerate(expectedOutputAsList) if x == 1]
        negativeIndices         = [i for i, x in enumerate(expectedOutputAsList) if x == -1]
        positiveX   = [self.inputVectors[j][0] for j in positiveIndices]
        positiveY   = [self.inputVectors[j][1] for j in positiveIndices]
        negativeX   = [self.inputVectors[j][0] for j in negativeIndices]
        negativeY   = [self.inputVectors[j][1] for j in negativeIndices]
        plt.scatter(positiveX , positiveY , color = "blue" , marker = "X" )
        plt.scatter(negativeX , negativeY , color = "red" , marker = "X" )

    def plotLoss(self) :
        plt.plot(range(len(self.loss_list)) , self.loss_list , "--")
        plt.show()

    def setInitialWeights(self) :
        self.setOfWeights = {}
        self.setOfWeightsForBiasTerm = {}
        for i in range(self.noOfHiddenLayers + 1) :
            noOfUnitsInNextLayer    = self.noOfUnitsInEachLayer[i+1]
            noOfUnitsInCurrentLayer = self.noOfUnitsInEachLayer[i]
            self.setOfWeightsForBiasTerm[i, i+1] = np.zeros(shape = (noOfUnitsInNextLayer, 1))
            self.setOfWeights[i, i+1] = np.random.normal(size = (noOfUnitsInNextLayer, noOfUnitsInCurrentLayer))

    def startAlgo(self) :
        self.loss_list = []
        j = 0
        avg_loss = 100
        noOfIterations = self.inputVectors.shape[0]//self.batchSize
        while j < self.noOfEpochs and avg_loss >= 0.01 :
            k = 0
            avg_loss = 0
            while k < noOfIterations :
                self.predictedOutput = self.forwardPass( self.inputVectors )
                loss = self.getLoss()
                self.loss_list.append( loss )
                batchIndexRange = range( self.batchSize*k , (self.batchSize*(k+1)))
                self.backpropagation( batchIndexRange )
                avg_loss += loss
                k += 1
            avg_loss = avg_loss/noOfIterations
            j += 1
#        print("list(zip(self.predictedOutput , self.expectedOutput)) : " , list(zip(self.predictedOutput , self.expectedOutput)))
        global start_time
        print("--- %s seconds ---" %(time.time()-start_time))

    def backpropagation(self , batchIndexRange) :
        self.calculateActivationFnDerivative()
        self.getWeightUpdationForOutputLayer( batchIndexRange )
        self.getWeightUpdationForHiddenLayers( batchIndexRange )
        self.updateWeights()

    def updateWeights(self) :
        for h in range(self.noOfHiddenLayers + 1) :
            self.setOfWeights[h,h+1]            -= self.learningRate * self.weightsDelta[h,h+1]
            self.setOfWeightsForBiasTerm[h,h+1] -= self.learningRate * self.biasWeightsDelta[h,h+1]

    def getWeightUpdationForHiddenLayers(self , batchIndexRange) :
        self.deltaContribution = self.deltaContribution.transpose((0,2,1))
        for h in range(self.noOfHiddenLayers, 0, -1) :
            weights                         = self.setOfWeights[h, h+1]
            activationDerivative            = self.activationDerivative[h][batchIndexRange].transpose((0,2,1))
            self.deltaContribution          = np.matmul(self.deltaContribution , weights * activationDerivative)
            activationPrevLayer             = self.activation[h-1][batchIndexRange]
            self.weightsDelta[h-1,h]        = np.mean(np.matmul(activationPrevLayer , self.deltaContribution) , axis=0).T
            self.biasWeightsDelta[h-1,h]    = np.mean(self.deltaContribution , axis=0).T

    def getWeightUpdationForOutputLayer(self , batchIndexRange) :
        self.weightsDelta               = {}
        self.biasWeightsDelta           = {}
        outputLayerIndex                = self.noOfHiddenLayers+1
        prevLayerToOutputLayerIndex     = outputLayerIndex-1
        predictedOutput                 = self.predictedOutput[batchIndexRange]
        expectedOutput                  = np.expand_dims(self.expectedOutput , axis=2)[batchIndexRange]
        lossDerivativeFn                = self.loss + "Derivative"
        lossDerivative                  = globals()[lossDerivativeFn](predictedOutput, expectedOutput)
        self.deltaContribution          = lossDerivative * self.activationDerivative[outputLayerIndex][batchIndexRange]
        activationAtPrevLayer           = self.activation[prevLayerToOutputLayerIndex][batchIndexRange]
        self.weightsDelta[prevLayerToOutputLayerIndex, outputLayerIndex]        = np.mean(np.matmul( self.deltaContribution , activationAtPrevLayer.transpose((0, 2, 1))) , axis=0)
        self.biasWeightsDelta[prevLayerToOutputLayerIndex, outputLayerIndex]    = np.mean(self.deltaContribution , axis=0)

    def calculateActivationFnDerivative(self) :
        self.activationDerivative = {}
        for h in range( self.noOfHiddenLayers+1 ) :
            activationDerivativeFn          = self.activationFnsForAllLayers[h] + "Derivative"
            self.activationDerivative[h+1]  = globals()[activationDerivativeFn]( self.weightedSums[h+1] )

    def getLoss(self) :
        lossFn = globals()[ self.loss ]
        expectedOutput  = np.expand_dims(self.expectedOutput , axis=2)
        return lossFn( self.predictedOutput , expectedOutput )

    def forwardPass(self , data) :
        self.activation     = {}
        self.weightedSums   = {}
        self.activation[0]  = np.expand_dims( data , axis = 2 )
        for h in range( self.noOfHiddenLayers+1 ) :
            self.weightedSums[h+1]      = np.matmul(self.setOfWeights[h,h+1] , self.activation[h]) + self.setOfWeightsForBiasTerm[h, h+1]
            activationFnForGivenLayer   = self.activationFnsForAllLayers[h]
            self.activation[h+1]        = globals()[activationFnForGivenLayer]( self.weightedSums[h+1] )
        outputLayerIndex        = self.noOfHiddenLayers + 1
        return self.activation[outputLayerIndex]

start_time = time.time()

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def tanh(x) :
    return np.tanh(x)

def l2_norm_squared(x, y) :
    return np.mean((x-y)**2)/2

def l2_norm_squaredDerivative(x, y) :
    noOfDataPts = x.shape[0]
    return (x-y)/noOfDataPts

def sigmoidDerivative(x) :
    return sigmoid(x)*(1-sigmoid(x))

def tanhDerivative(x) :
    return (1-tanh(x) ** 2)

def ellipseFn(x , a , b) :
    return (b/a)*((a**2-x**2)**0.5)

# CREATING LINEARLY SEPARABLE DATA
def runForLinearlySeparableData() :
    args = {}
    noOfDataPts = 80
    shuffledIndices = np.random.permutation( noOfDataPts )
    args["inputVectors"]                = (np.concatenate((np.random.normal(loc=10, size=[40, 2]), np.random.normal(loc=20, size=[40, 2]))) / 20)[shuffledIndices]
    args["expectedOutput"]              = (np.concatenate((np.ones(shape=(40, 1)), -np.ones(shape=(40, 1)))))[shuffledIndices]
    args["noOfEpochs"]                  = 100000
    args["activationFnsForAllLayers"]   = ["tanh"]*3
    args["noOfUnitsInEachLayer"]        = [ 2, 6, 6, 1 ]
    args["lossFn"]                      = "l2_norm_squared"
    args["learningRate"]                = 0.1
    args["batchSize"]                   = 1
    stochasticGradientObj               = stochasticGradient( args )
    stochasticGradientObj.start()

# CREATING TWO CONCENTRIC ELLIPSES
def runForEllipseData() :
    inputs = {}
    r = [ 2 , 5 ]
    h = 0.2
    inputVectorsList = []
    expectedOutput = []
    for i in r :
        t = (i-(-i))/h
        x = np.linspace(-i , i , t)
        vectorizedEllipseFn = np.vectorize( ellipseFn )
        y = vectorizedEllipseFn( x , i , i )
        for j in range(len(x)):
            inputVectorsList += [(x[j], -y[j]), (x[j], y[j])]
            if i == 2 :
                expectedOutput.append([1])
                expectedOutput.append([1])
            else:
                expectedOutput.append([-1])
                expectedOutput.append([-1])
    perm = np.random.permutation(140)
    inputs["inputVectors"]              = np.array(inputVectorsList)[perm]/5
    inputs["expectedOutput"]            = np.array(expectedOutput)[perm]
    inputs["noOfEpochs"]                = 200000
    inputs["activationFnsForAllLayers"] = ["tanh" , "tanh" ]
    inputs["noOfUnitsInEachLayer"]      = [ 2 , 3 , 1 ]
    inputs["lossFn"]                    = "l2_norm_squared"
    inputs["learningRate"]              = 0.5
    inputs["batchSize"]                 = 140
    stochasticGradientObj               = stochasticGradient( inputs )
    stochasticGradientObj.start()

# CREATING XOR DATA
def runForXORdata() :
    inputs = {}
    inputs["inputVectors"]              = np.array([[0,0] , [0,1] , [1,1] , [1,0]])
    inputs["expectedOutput"]            = np.array([[-1],[1],[-1],[1]])
    inputs["noOfEpochs"]                = 200000
    inputs["activationFnsForAllLayers"] = ["tanh" , "tanh" ]
    inputs["noOfUnitsInEachLayer"]      = [ 2 , 3 , 1 ]
    inputs["lossFn"]                    = "l2_norm_squared"
    inputs["learningRate"]              = 0.05
    inputs["batchSize"]                 = 1
    stochasticGradientObj               = stochasticGradient( inputs )
    stochasticGradientObj.start()

runForLinearlySeparableData()
runForEllipseData()
runForXORdata()
