import numpy as np
import cv2
import random
from scipy import optimize

def main():

    HIDDEN_LAYER_NEURONS = 50
    OUTPUT_NEURONS = 10
    file_data = np.loadtxt("../../train.csv", dtype=np.uint8, skiprows=1, delimiter=",")  # Load the data
    y = file_data[:, 0]  # Extract the labels
    x = file_data[:, 1:] # Extract the unrolled images
    y_matrix = np.zeros((y.shape[0], OUTPUT_NEURONS))   # Initialize Matrix to hold label vectors
    for i in range(0, y.shape[0]):  # Apply one hot on the vectors
        y_matrix[i, y[i]] = 1
    """
        Initialize weights using Gaussian distribution
        Theta1 of dimensions equal to # of Input neurons + 1
        and # of hidden neurons
        Theta2 of dimensions equal to # of Output neurons
        and # of hidden neurons + bias neuron
    """
    Theta1 = np.random.normal(scale=0.01, size=(HIDDEN_LAYER_NEURONS, x.shape[1]))
    Theta1 = np.c_[np.ones((Theta1.shape[0], 1)), Theta1]
    Theta2 = np.random.normal(scale=0.01, size=(y_matrix.shape[1], HIDDEN_LAYER_NEURONS))
    Theta2 = np.c_[np.ones((Theta2.shape[0], 1)), Theta2]
    Theta1_unrolled = Theta1.reshape((1, Theta1.shape[0] * Theta1.shape[1]))
    Theta2_unrolled = Theta2.reshape((1, Theta2.shape[0] * Theta2.shape[1]))
    Theta = np.concatenate((Theta1_unrolled, Theta2_unrolled), axis=1)

    # visualize(x, 10)   # Visualize some random samples
    visualizeWithPredict(x, ((Theta1.shape), (Theta2.shape)), 10)
    # opt = optimize.fmin_cg(tempCost, Theta, maxiter=100, args=(x, y_matrix, Theta1.shape, Theta2.shape), callback=saveResults, retall=True, fprime=tempGrad)
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

def costFunction(Theta, X, Y, th1_shape, th2_shape):
    Theta1 = Theta[0:(th1_shape[0] * th1_shape[1])]
    Theta1 = Theta1.reshape(th1_shape)
    Theta2 = Theta[(th1_shape[0] * th1_shape[1]):]
    Theta2 = Theta2.reshape(th2_shape)
    Hypoths, Theta1_grad, Theta2_grad = forwardProp(X, Y, Theta1, Theta2)
    J = (sum(sum( (-1*Y)*np.log(Hypoths) - (1-Y)*np.log(1-Hypoths) )))/X.shape[0]
    Theta_grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)
    return (J, Theta_grad)

def forwardProp(X, Y, Theta1, Theta2):
    a1 = np.c_[np.ones((X.shape[0], 1)), X]
    

    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones((a2.shape[0], 1)), a2]

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    d3 = a3 - Y
    d2 = (d3.dot(Theta2[:, 1:])) * sigmoid_gradient(z2)

    Delta1 = d2.T.dot(a1)
    Delta2 = d3.T.dot(a2)

    Theta1_grad = Delta1/(X.shape[0])
    Theta2_grad = Delta2/(X.shape[0])

    return (a3, Theta1_grad, Theta2_grad)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1-sigmoid(x))
def saveResults(opt):
    np.savetxt("params.txt", opt)

def tempCost(Theta, X, Y, th1_shape, th2_shape):
    return costFunction(Theta, X, Y, th1_shape, th2_shape)[0]

def tempGrad(Theta, X, Y, th1_shape, th2_shape):
    return costFunction(Theta, X, Y, th1_shape, th2_shape)[1]

def visualizeWithPredict(X, th_shapes, iterations):
    weights = np.loadtxt("params.txt")
    # X = np.loadtxt("../../test.csv", dtype=np.uint8, skiprows=1, delimiter=",")
    Theta1_shape = th_shapes[0]
    Theta2_shape = th_shapes[1]
    Theta1 = weights[0:Theta1_shape[0]*Theta1_shape[1]]
    Theta2 = weights[Theta1_shape[0]*Theta1_shape[1]:]
    Theta1 = Theta1.reshape(Theta1_shape)
    Theta2 = Theta2.reshape(Theta2_shape)
    for i in range(0, iterations):
        row = X[random.randint(0, X.shape[0] - 1)]
        img = row.reshape((28,28))
        x = row.reshape((1, row.shape[0]))
        Y = np.zeros((1, 10))
        predictions = forwardProp(x, Y, Theta1, Theta2)[0]
        cv2.imshow("Image", img)
        print predictions.argmax(axis=1)
        print predictions
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()