
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
	global outputVec
	outputVec[0] = activation(np.dot(weightMats[0], inputVec[inputNum]))
	for i in range(1, len(hiddenLayers)):
		outputVec[i] = activation(np.dot(weightMats[i], outputVec[i-1]))

def propagateBackward(inputNum):
	global deltaVec
	global costFunction
	errorVec = np.subtract(desiredOutputVec[-1], outputVec[-1])
	costFunction += np.sum(errorVec**2)
	deltaVec = range(len(hiddenLayers))
	deltaVec[-1] = np.multiply(errorVec, activation(outputVec[-1]))
	for i in range(len(hiddenLayers) - 2, -1, -1):
		deltaVec[i] = np.multiply(np.dot(deltaVec[i+1], weightMats[i+1]), activationDer(outputVec[i]))

def adjustWeights():
	global prevWeights
	currWeights = range(len(weightMats))
	for i in range(1, len(hiddenLayers)):
		currWeights[i] = np.copy(weightMats[i])
		deltaVec[i].shape = (len(deltaVec[i]), 1)
		weightMats[i] = np.add(np.add(currWeights[i], np.multiply(np.subtract(prevWeights[i], currWeights[i]), momentum)), np.multiply(np.multiply(deltaVec[i], outputVec[i-1]), 2 * learningRate))
		prevWeights = currWeights

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
			adjustWeights()

		if (epoch % errorSavingStep == 0):
			fileObject.write(str(costFunction) + '\n')
			
		epoch += 1

	fileObject.close()
	
def testingMode():
	for inputNum in range(len(inputVec)):
		propagateForward(inputNum)

randMinVal = -1.0
randMaxVal = 1.0
generations = 20
randomOrder = 1
momentum = 0.7
learningRate = 0.01
errorSavingStep = 2

costFunction = 0
data = readDataFromFile('data/iris.data')
inputVec = data[0]
desiredOutputVec = data[1]

hiddenLayers = [3, len(desiredOutputVec[0])]
weightMats = list()
weightMats.append(np.random.uniform(low=randMinVal, high=randMaxVal, size=(hiddenLayers[0], len(inputVec[0]))))
for i in range(1, len(hiddenLayers)):
	weightMats.append(np.random.uniform(low=randMinVal, high=randMaxVal, size=(hiddenLayers[i], len(weightMats[i-1]))))
prevWeights = range(len(weightMats))
outputVec = range(len(hiddenLayers))

learningMode()
testingMode()

plotCostFunction('out/costFunction')
