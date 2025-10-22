from MyNeuralNetwork import MyNeuralNetwork

n = MyNeuralNetwork(3, 3, 3, 0.3)
a = n.query([1.0, 0.5, -1.5])

print(a)