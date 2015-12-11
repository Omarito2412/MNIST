import numpy as np
import cv2
import random
from scipy import optimize

class nn:
    def __init__(self, neurons, mode="train"):

        self.HIDDEN_LAYER_NEURONS = neurons
        self.OUTPUT_NEURONS = 10
        if (mode == "test"):
            self.visualizeWithPredict(10)
            return
        file_data = np.loadtxt("../../train.csv", dtype=np.uint8, skiprows=1, delimiter=",")  # Load the data
        y = file_data[:, 0]  # Extract the labels
        x = file_data[:, 1:] # Extract the unrolled images
        y_matrix = np.zeros((y.shape[0], self.OUTPUT_NEURONS))   # Initialize Matrix to hold label vectors
        for i in range(0, y.shape[0]):  # Apply one hot on the vectors
            y_matrix[i, y[i]] = 1
        """
            Initialize weights using Gaussian distribution
            Theta1 of dimensions equal to # of Input neurons + 1
            and # of hidden neurons
            Theta2 of dimensions equal to # of Output neurons
            and # of hidden neurons + bias neuron
        """
        Theta1 = np.random.normal(scale=0.01, size=(self.HIDDEN_LAYER_NEURONS, x.shape[1]))
        Theta1 = np.c_[np.ones((Theta1.shape[0], 1)), Theta1]
        Theta2 = np.random.normal(scale=0.01, size=(y_matrix.shape[1], self.HIDDEN_LAYER_NEURONS))
        Theta2 = np.c_[np.ones((Theta2.shape[0], 1)), Theta2]
        Theta1_unrolled = Theta1.reshape((1, Theta1.shape[0] * Theta1.shape[1]))
        Theta2_unrolled = Theta2.reshape((1, Theta2.shape[0] * Theta2.shape[1]))
        Theta = np.concatenate((Theta1_unrolled, Theta2_unrolled), axis=1).ravel()
        Lambda = 0.01
        self.costFunction(Theta, x, y_matrix, Theta1.shape, Theta2.shape, Lambda)
        # visualize(x, 10)   # Visualize some random samples
        # visualizeWithPredict(x, ((Theta1.shape), (Theta2.shape)), 10)
        opt = optimize.fmin_cg(self.tempCost, Theta, maxiter=200, args=(x, y_matrix, Theta1.shape, Theta2.shape, Lambda), callback=self.saveResults, fprime=self.tempGrad)
        # print costFunction(x, y_matrix, Theta1, Theta2)

    """
        Visualize #iterations of random samples
        from the matrix rows
    """
    def visualize(rows, iterations):
        for i in range(0, iterations):
            row = rows[random.randint(0, rows.shape[0] - 1)]
            img = row.reshape((28,28))
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    """
        The cost function as defined
    """
    def costFunction(self, Theta, X, Y, th1_shape, th2_shape, Lambda):
        Theta1 = Theta[0:(th1_shape[0] * th1_shape[1])]
        Theta1 = Theta1.reshape(th1_shape)
        Theta2 = Theta[(th1_shape[0] * th1_shape[1]):]
        Theta2 = Theta2.reshape(th2_shape)
        Hypoths, Theta1_grad, Theta2_grad = self.forwardProp(X, Y, Theta1, Theta2, Lambda)
        costP = (-1*Y)*np.log(Hypoths)
        costN = (1-Y)*np.log(1-Hypoths)
        regulznParam = (Lambda/(2*X.shape[0])) * (np.sum(Theta1**2) + np.sum(Theta2**2))
        self.J = (np.sum( costP - costN ))/X.shape[0]
        self.J = self.J + regulznParam
        self.Theta_grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)
        return (self.J, self.Theta_grad)


    """
        Perform forward propagation on the network
        and backward propagation, this returns the
        cost of the network and the gradients to
        the parameters
    """
    def forwardProp(self, X, Y, Theta1, Theta2, Lambda):
        a1 = np.c_[np.ones((X.shape[0], 1)), X]     # Prepend X with a column of ones (Bias term)
        

        z2 = a1.dot(Theta1.T)   # Compute Z2
        a2 = self.sigmoid(z2)   # Activation of Z2
        a2 = np.c_[np.ones((a2.shape[0], 1)), a2]   # Prepend a2 with a column of ones (Bias term)

        z3 = a2.dot(Theta2.T)   # Computed Z3
        a3 = self.sigmoid(z3)   # Activation of Z3

        """ Back propagation starts here """

        d3 = a3 - Y     # delta of layer 3
        d2 = (d3.dot(Theta2[:, 1:])) * self.sigmoid_gradient(z2)    # Compute delta of layer 2

        Delta1 = d2.T.dot(a1) # Compute Delta 1  
        Delta1 = np.divide(Delta1, X.shape[0]) # Divide by m
        # Add regularized term
        Delta1 = Delta1 + (Lambda * np.c_[np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]])/X.shape[0]

        Delta2 = d3.T.dot(a2)   # Compute Delta 2
        Delta2 = np.divide(Delta2, X.shape[0])  # Divide by m
        # Add regularized term
        Delta2 = Delta2 + (Lambda * np.c_[np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]])/X.shape[0]

        Theta1_grad = Delta1/(X.shape[0])   # The computed Theta1 gradient
        Theta2_grad = Delta2/(X.shape[0])   # The computed Theta2 gradient

        return (a3, Theta1_grad, Theta2_grad)

    """
        1 / (1+e^-x)
    """
    def sigmoid(self, x):
        return np.divide(1., (1 + np.exp(-x)))

    """
        Gradient of sigmoid
    """
    def sigmoid_gradient(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    """
        Callback function during each iteration to
        save the results found
    """
    def saveResults(self, opt):
        np.savetxt("params.txt", opt)

    """
        Had to resort to this to overcome the
        single allowed return value from costFunction
        into fmin_cg
    """
    def tempCost(self, Theta, X, Y, th1_shape, th2_shape, Lambda):
        self.costFunction(Theta, X, Y, th1_shape, th2_shape, Lambda)[0]
        return self.J

    """
        Wasteful function that I must use to return
        the gradient from Forward Prop to the fmin_cg
        function, since it only accepts 1 return value
    """
    def tempGrad(self, Theta, X, Y, th1_shape, th2_shape, Lambda):
        return self.Theta_grad

    """
        Visualize some random samples and
        predict their value
    """
    def visualizeWithPredict(self, iterations):
        weights = np.loadtxt("params.txt")
        X = np.loadtxt("../../tiny.csv", dtype=np.uint8, skiprows=1, delimiter=",")
        X = X[:, 0:X.shape[1]-1]
        Theta1_shape = (self.HIDDEN_LAYER_NEURONS, X.shape[1] + 1)
        Theta2_shape = (self.OUTPUT_NEURONS, self.HIDDEN_LAYER_NEURONS + 1)
        Theta1 = weights[0:Theta1_shape[0]*Theta1_shape[1]]
        Theta2 = weights[Theta1_shape[0]*Theta1_shape[1]:]
        Theta1 = Theta1.reshape(Theta1_shape)
        Theta2 = Theta2.reshape(Theta2_shape)
        for i in range(0, iterations):
            row = X[random.randint(0, X.shape[0] - 1)]
            img = row.reshape((28,28))
            x = row.reshape((1, row.shape[0]))
            Y = np.zeros((1, 10))
            predictions = self.forwardProp(x, Y, Theta1, Theta2, 0)[0]
            cv2.imshow("Image", img)
            print predictions.argmax(axis=1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

Network = nn(200, "train")