
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def readDataFromFile(filename):
	newInput = list()
	newOutput = list()
	fileObject = open(filename, 'r')
	text = fileObject.read()
	fileObject.close()
	lines = text.split('\n')
	random.shuffle(lines)
	for i in range(len(lines)):
		if lines[i] == '':
			continue
		line = lines[i].split('->')
		newInput.append(np.array([float(j) for j in line[0].split(',')]))
		newOutput.append(np.array([float(j) for j in line[1].split(',')]))
	return [newInput, newOutput]

def plotCostFunction(filename):
	fileObject = open(filename, 'r')
	plt.plot([float(errorSavingStep*i) for i in range(1, 1 + int(generations / errorSavingStep))], [float(i) for i in fileObject.read().strip().split('\n')])
	fileObject.close()
	plt.ylabel('cost function')
	plt.xlabel('epoch')
	plt.show()

def activation(x):
	if x < -100.0:
		return 0.0
	return 1.0/(1.0+math.exp(-x))
activation = np.vectorize(activation)

def activationDer(x):
	return x * (1.0 - x)
activationDer = np.vectorize(activationDer)

def propagateForward(inputNum):
	global outputVecs
	outputVecs[0] = activation(np.add(np.dot(weightMats[0], inputVec[inputNum]), biasVecs[0]))
	for i in range(1, len(hiddenLayers)):
		outputVecs[i] = activation(np.add(np.dot(weightMats[i], outputVecs[i-1]), biasVecs[i]))

def propagateBackward(inputNum):
	global deltaVecs
	global costFunction
	errorVec = np.subtract(desiredOutputVec[inputNum], outputVecs[-1])
	costFunction += np.sum(errorVec**2)
	deltaVecs = range(len(hiddenLayers))
	deltaVecs[-1] = np.multiply(errorVec, activation(outputVecs[-1]))
	for i in range(len(hiddenLayers) - 2, -1, -1):
		deltaVecs[i] = np.multiply(np.dot(deltaVecs[i+1], weightMats[i+1]), activationDer(outputVecs[i]))

def adjustWeights(inputNum):
	global deltaVecs
	global prevWeightMats
	global prevBiasVecs
	currWeightMats = range(len(weightMats))
	currBiasVecs = range(len(biasVecs))
	for i in range(len(hiddenLayers)):
		currWeightMats[i] = np.copy(weightMats[i])
		currBiasVecs[i] = np.copy(biasVecs[i])
		deltaVecs[i].shape = (len(deltaVecs[i]), 1)
		if i == 0:
			weightMats[i] = np.add(np.add(currWeightMats[i], np.multiply(np.subtract(prevWeightMats[i], currWeightMats[i]), momentum)), np.multiply(np.multiply(deltaVecs[i], inputVec[inputNum]), 2 * learningRate))
		else:
			weightMats[i] = np.add(np.add(currWeightMats[i], np.multiply(np.subtract(prevWeightMats[i], currWeightMats[i]), momentum)), np.multiply(np.multiply(deltaVecs[i], outputVecs[i-1]), 2 * learningRate))
		deltaVecs[i].shape = (len(deltaVecs[i]))
		biasVecs[i] = np.add(np.add(currBiasVecs[i], np.multiply(np.subtract(prevBiasVecs[i], currBiasVecs[i]), momentum)), np.multiply(deltaVecs[i], 2 * learningRate))
		prevWeightMats = currWeightMats
		prevBiasVecs = currBiasVecs

def learningMode():
	global costFunction
	numbers = [i for i in range (len(inputVec))]
	inputNum = 0
	randNoRepeat = 0

	fileObject = open('out/costFunction', 'w+')

	epoch = 0
	while epoch < generations:
		costFunction = 0
		for i in range(len(inputVec)):
			if(randomOrder):
				randNoRepeat = random.randint(0, len(numbers) - i - 1)
				inputNum = numbers[randNoRepeat]
				numbers[randNoRepeat] = numbers[len(numbers) - i - 1]
				numbers[len(numbers) - i - 1] = inputNum
			else:
				inputNum = i
			propagateForward(inputNum)
			propagateBackward(inputNum)
			adjustWeights(inputNum)

		if (epoch % errorSavingStep == 0):
			fileObject.write(str(costFunction) + '\n')
			
		epoch += 1

	fileObject.close()
	
def testingMode():
	for inputNum in range(len(inputVec)):
		propagateForward(inputNum)
		expectedClass = np.argmax(desiredOutputVec[inputNum])
		resultClass = np.argmax(outputVecs[-1])
		confusionMat[expectedClass][resultClass] += 1

randMinVal = -1.0
randMaxVal = 1.0
generations = 20
randomOrder = 1
momentum = 0.7
learningRate = 0.01
errorSavingStep = 1
biasFlag = 1

costFunction = 0
data = readDataFromFile('data/iris.data')
inputVec = data[0]
desiredOutputVec = data[1]

hiddenLayers = [10, len(desiredOutputVec[0])]
weightMats = list()
weightMats.append(np.random.uniform(low=randMinVal, high=randMaxVal, size=(hiddenLayers[0], len(inputVec[0]))))
for i in range(1, len(hiddenLayers)):
	weightMats.append(np.random.uniform(low=randMinVal, high=randMaxVal, size=(hiddenLayers[i], len(weightMats[i-1]))))
prevWeightMats = range(len(weightMats))
biasVecs = list()
if biasFlag:
	for i in range(len(hiddenLayers)):
		biasVecs.append(np.random.uniform(low=randMinVal, high=randMaxVal, size=hiddenLayers[i]))
else:
	for i in range(len(hiddenLayers)):
		biasVecs.append(np.zeros(hiddenLayers[i]))
prevBiasVecs = range(len(biasVecs))
outputVecs = range(len(hiddenLayers))
confusionMat = np.zeros((len(desiredOutputVec[0]), len(desiredOutputVec[0])))

learningMode()
testingMode()

print confusionMat
plotCostFunction('out/costFunction')
