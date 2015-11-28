import numpy as np
import cv2
import random

def main():

    HIDDEN_LAYER_NEURONS = 50
    OUTPUT_NEURONS = 10
    file_data = np.loadtxt("../../train.csv", dtype=np.uint8, skiprows=1, delimiter=",")  # Load the data
    y = file_data[:, 0]  # Extract the labels
    x = file_data[:, 1:] # Extract the unrolled images
    # visualize(x, 10)   # Visualize some random samples
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
    Theta1 = np.random.normal(scale=0.01, size=(HIDDEN_LAYER_NEURONS, x.shape[1] + 1))
    Theta2 = np.random.normal(scale=0.01, size=(y_matrix.shape[1], HIDDEN_LAYER_NEURONS + 1))
    costFunction(x, y_matrix, Theta1, Theta2)


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

def costFunction(X, Y, Theta1, Theta2):
    Hypoths = forwardProp(X, Y, Theta1, Theta2)
    
    J = (1/X.shape[0]) * sum(sum( (-1*Y)*np.log(Hypoths) - (1-Y)*np.log(1-Hypoths) ))
    print J

def forwardProp(X, Y, Theta1, Theta2):
    a1 = np.c_[np.ones((X.shape[0], 1)), X]

    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones((a2.shape[0], 1)), a2]

    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    return a3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

main()