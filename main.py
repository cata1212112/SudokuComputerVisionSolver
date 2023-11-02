import sys

import numpy as np

from Loader import *
from Preprocessing import *
from imports import plt
from Constants import *
from Classifier import *
from Sudoku import *

def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def validate(game, cells):
    for i in range(9):
        for j in range(9):
            isEmpty = Classifier.isEmpty(cells[i, j])
            assert (isEmpty == False and game[i,j] != 0) or (isEmpty == True and game[i,j] == 0)


def splitByNumber(games, allCells):
    d = {}
    for i in range(1, 10):
        d[i] = []

    for game, cells in zip(games, allCells):
        for i in range(9):
            for j in range(9):
                isEmpty = Classifier.isEmpty(cells[i, j])
                if not isEmpty:
                    grayScaledCell = cv.cvtColor(cells[i, j], cv.COLOR_BGR2GRAY)
                    d[game[i][j]].append(grayScaledCell)

    return d

def getTemplate(cell):
    test = cell.copy()
    initial = test.copy()
    test = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
    _, test = cv.threshold(test, 127, 255, cv.THRESH_BINARY_INV)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 15))
    dilation = cv.dilate(test, rect_kernel, iterations=1)
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x, y, w, h = cv.boundingRect(contours[0])
    roi = initial[y:y+h, x:x+w].copy()
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    return roi

# index = 9
#
Xtrain, Ytrain = Loader(trainImagesPath).load()

Xtest, _ = Loader(testImagesPath).load()

games = []
allCells = []

print("Testing train")
for i in range(20):
    cells, img = Preprocess(Xtrain[i]).computeIntersectionPoints()
    sudoku = Sudoku(Ytrain[i])
    validate(sudoku.game, cells)
    games.append(sudoku.game)
    allCells.append(cells)

classes = splitByNumber(games, allCells)

print("Testing test")
for index in [19]:
    processing = Preprocess(Xtest[index])
    cells, img = processing.computeIntersectionPoints()
    showImage(Xtest[index])

    currentSudokuGame = np.zeros((9, 9), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            if not Classifier.isEmpty(cells[i, j]):
                template = getTemplate(cells[i,j])

                maximumValue = -1
                actualClass = -1

                for c in range(1, 10):
                    for img in classes[c]:
                        res = cv.matchTemplate(img, template, cv.TM_CCOEFF)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        if max_val > maximumValue:
                            maximumValue = max_val
                            actualClass = c

                currentSudokuGame[i, j] = actualClass


    solution = Sudoku.solveSudoku(currentSudokuGame)
    # print(currentSudokuGame)
    # print(solution)
    toWrite = np.zeros((9, 9), dtype=np.uint8)
    toWrite = currentSudokuGame != 0

    img = processing.getAugmentedImage(solution, toWrite)
    showImage(img)