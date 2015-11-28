import numpy as np
import cv2
import random

def main():
    file_data = np.loadtxt("../../train.csv", dtype=np.uint8, skiprows=1, delimiter=",")  # Load the data
    y = file_data[:, 0]  # Extract the labels
    x = file_data[:, 1:] # Extract the unrolled images
    # visualize(x, 10)   # Visualize some random samples
    y_matrix = np.zeros((y.shape[0], y.shape[0]))   # Initialize Matrix to hold label vectors
    for i in range(0, y.shape[0]):  # Apply one hot on the vectors
        y_matrix[i, y[i]] = 1

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

main()