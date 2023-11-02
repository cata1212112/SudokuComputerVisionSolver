from imports import cv
from imports import np

class Classifier:
    def __init__(self):
        pass

    @classmethod
    def isEmpty(self, cell):
        processedImage = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        _, processedImage = cv.threshold(processedImage, 127, 255, cv.THRESH_BINARY_INV)
        return np.max(processedImage) == 0

