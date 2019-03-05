# back-propagation-algo-implementation
Back Propagation algorithm implemented from scratch using python. The program accepts number of hidden layers, number of neurons (should be given as a list consisting of number of neurons for all layers starting from input layer followed by hidden layers and output layer), activation functions (provided the functions and their derivatives are defined in the code, the functions should be given in the form of a list with each function for a hidden layer followed by one for the output layer), loss function (provided the function and its derivative are defined in the code), number of epochs in addition to the data, labels, learning rate and batch size as arguments. The following three types of data are considered to test if the algorithm is working properly: -

1. Linearly separable data is generated using normal distribution with different means.
2. Two concentric ellipses are generated using different measures for semi-major and semi-minor axes but same center.
3. XOR data

There are two output images associated with each of the above cases: -
(a) The error plot, which plots the error against the number of iterations till the error crosses a certain threshold
(b) The classifier plot, with crosses denoting data points
