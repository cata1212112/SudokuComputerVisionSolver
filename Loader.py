from imports import os
from imports import cv
from Constants import imageType


class Loader:
    def __init__(self, path):
        self.images = []
        self.path = path
        self.sudokus = []
    def load(self):
        images = os.listdir(self.path)
        for image in images:
            if image[-3:] == imageType:
                img = cv.imread(os.path.join(self.path, image))
                self.images.append(img)
            elif image[-3:] == "txt" and "bonus" in image:
                sudoku = []
                with open(os.path.join(self.path, image)) as f:
                    for line in f:
                        sudoku.append(line)
                self.sudokus.append(sudoku)
        return self.images, self.sudokus
