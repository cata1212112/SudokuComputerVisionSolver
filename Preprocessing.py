from imports import cv
from imports import np
from imports import math

class Preprocess:
    def __init__(self, image):
        self.originalImage = image
        self.sudokuGrid = None
        h, w, ch = image.shape
        border_size = 10
        cropped_height = h - 2 * border_size
        cropped_width = w - 2 * border_size
        cropped_image = image[border_size: border_size + cropped_height, border_size: border_size + cropped_width]
        self.image = cropped_image
        self.cellsPositions = [[() for _ in range(9)] for _ in range(9)]
        self.inverse_M = None


    def binarize(self):
        processedImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        _, processedImage = cv.threshold(processedImage, 127, 255, cv.THRESH_BINARY_INV)
        return processedImage

    def getContours(self):
        processedImage = self.binarize()
        contours, hierarchy = cv.findContours(processedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        maximumPerimeter = -1
        sudokuGrid = None
        corners = None

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                if perimeter > maximumPerimeter:
                    corners = approx
                    maximumPerimeter = perimeter
                    sudokuGrid = contour


        corners = cv.convexHull(corners)
        index_min = np.argmin(np.array([c[0][0] ** 2 + c[0][1] ** 2 for c in corners]))
        corners = np.concatenate((corners[index_min:], corners[:index_min]))
        processedImage = cv.drawContours(self.image.copy(), [sudokuGrid], -1, (0, 255, 0), 3)
        return corners, processedImage

    def getGridAsSquare(self):
        corners, _ = self.getContours()

        src = np.float32([c[0] for c in corners])
        # print(corners)
        width = max(int(np.round(np.linalg.norm(np.array([[corners[0][0][0] - corners[1][0][0], corners[0][0][1] - corners[1][0][1]]])))), int(np.round(np.linalg.norm(np.array([[corners[2][0][0] - corners[3][0][0], corners[2][0][1] - corners[3][0][1]]])))))
        height = max(int(np.round(np.linalg.norm(np.array([[corners[0][0][0] - corners[3][0][0], corners[0][0][1] - corners[3][0][1]]])))), int(np.round(np.linalg.norm(np.array([[corners[1][0][0] - corners[2][0][0], corners[1][0][1] - corners[2][0][1]]])))))
        # print(width, height)
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv.getPerspectiveTransform(src, dst_pts)
        self.inverse_M = cv.getPerspectiveTransform(dst_pts, src)
        sudoku = cv.warpPerspective(self.image, M, (width, height))

        self.sudokuGrid = sudoku
        self.saveSudokuGrid = sudoku
        return width, height, sudoku

    def getCells(self):
        width, height, sudokuGrid = self.getGridAsSquare()

        border_size = 10

        h, w, ch = sudokuGrid.shape
        bordered_image = np.zeros((h + 2 * border_size, w + 2 * border_size, ch), dtype=np.uint8)
        bordered_image[border_size:border_size + height, border_size:border_size + width] = sudokuGrid

        bordered_image_forInitial = np.zeros((h + 2 * border_size, w + 2 * border_size, ch), dtype=np.uint8)
        bordered_image_forInitial[border_size:border_size + height, border_size:border_size + width] = self.sudokuGrid
        self.sudokuGrid = bordered_image_forInitial

        grid = np.zeros(bordered_image.shape, dtype=np.uint8)

        processedImage = cv.cvtColor(bordered_image, cv.COLOR_BGR2GRAY)
        processedImage = cv.GaussianBlur(processedImage, (3, 3), 0)
        _, processedImage = cv.threshold(processedImage, 127, 255, cv.THRESH_BINARY_INV)
        processedImage = cv.Canny(processedImage, threshold1=50, threshold2=100)
        contours, hierarchy = cv.findContours(processedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnt = 0
        minPerimeter = -1
        aux = []
        dict = {}

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                aux.append(perimeter)
                if perimeter not in dict:
                    dict[perimeter] = 1
                else:
                    dict[perimeter] += 1
                minPerimeter = min(minPerimeter, perimeter)
                cnt += 1

        aux = sorted(aux)

        processedImage = cv.drawContours(grid, contours, -1, (255, 255, 255), 3)
        return processedImage


    def getEdges(self, img):
        processedImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, processedImage = cv.threshold(processedImage, 127, 255, cv.THRESH_BINARY_INV)
        processedImage = cv.Canny(processedImage, threshold1=50, threshold2=100)
        return processedImage

    def drawLines(self, lines, col, img):
        newImage = img.copy()
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))

            newImage = cv.line(img, pt1, pt2, col, 3, cv.LINE_AA)

        return newImage

    def cleanLines(self, lines):
        i = 0
        cleanLines = []
        while i < len(lines):
            j = i
            rho = lines[i][0][0]
            group = []
            while j < len(lines) and lines[j][0][0] - rho < 50:
                group.append(lines[j])
                rho = lines[j][0][0]
                j += 1
            cleanLines.append(group[len(group) // 2])
            i = j - 1
            i += 1

        return cleanLines
    def getLines(self):
        sudoku = self.getCells()
        canny = self.getEdges(sudoku)
        lines = cv.HoughLines(canny, 1, 3 * np.pi / 180, threshold=300)

        horizontalLines = []
        verticalLines = []
        if lines is not None:
            for i in range(len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)


                if theta < np.pi / 10 or theta > np.pi - np.pi / 10:
                    verticalLines.append(lines[i])
                elif np.abs(theta - np.pi / 2) < np.pi / 10:
                    horizontalLines.append(lines[i])
                else:
                    pass



        verticalLines = np.array(sorted(verticalLines, key=lambda line : line[0][0]))
        horizontalLines = np.array(sorted(horizontalLines, key=lambda line : line[0][0]))

        # print("Linii")
        # print(len(verticalLines), len(horizontalLines))

        cleanVerticalLines = self.cleanLines(verticalLines)
        cleanHorizontalLines = self.cleanLines(horizontalLines)


        assert len(cleanHorizontalLines) == 10 and len(cleanVerticalLines) == 10, "Actual values {}, {}".format(len(cleanHorizontalLines), len(cleanVerticalLines))

        imageWithLines = self.drawLines(cleanVerticalLines, (255, 0, 0), self.sudokuGrid)
        imageWithLines = self.drawLines(cleanHorizontalLines, (0, 0, 255), imageWithLines)

        return cleanVerticalLines, cleanHorizontalLines, imageWithLines

    def intersect(self, line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return (x0, y0)

    def distance(self, pointA, pointB):
        return np.linalg.norm(np.array([pointA[0] - pointB[0], pointA[1] - pointB[1]]))

    def computeIntersectionPoints(self):
        verticalLines, horizontalLines, _ = self.getLines()

        imageWithPoints = self.sudokuGrid.copy()

        gridVerticalLines = verticalLines
        gridHorizontalLines = horizontalLines
        allIntersections = []

        for vertical in gridVerticalLines:
            intersections = []
            for horizontal in gridHorizontalLines:
                point = np.array(self.intersect(vertical, horizontal))
                intersections.append(point)

            allIntersections.append(intersections)
            for point in intersections:
                imageWithPoints = cv.circle(imageWithPoints, point, 30, (255, 0, 0), thickness=-1)

        cnt = 0

        cells = np.array([[None for _ in range(len(allIntersections) - 1)] for _ in range(len(allIntersections) - 1)])


        for i in range(len(allIntersections) - 1):
            for j in range(len(allIntersections[i]) - 1):
                cnt += 1
                resizeMargin = 30
                UL = allIntersections[i][j]
                UL = (UL[0] + resizeMargin, UL[1] + resizeMargin)

                BR = allIntersections[i+1][j+1]
                BR = (BR[0] - resizeMargin, BR[1] - resizeMargin)


                cells[j, i] = self.sudokuGrid[UL[1] : BR[1], UL[0]: BR[0]].copy()
                self.cellsPositions[j][i] = (UL, BR)

                imageWithPoints = cv.rectangle(imageWithPoints, UL, BR, (255, 0, 0), 5)

        assert cnt == 81, "Not found all cells"
        return cells, imageWithPoints

    def drawResult(self, sudokuGame, whereToWrite):
        copie = self.saveSudokuGrid.copy()
        for i in range(0, 9):
            for j in range(0, 9):
                if not whereToWrite[i, j]:
                    UL = self.cellsPositions[i][j][0]
                    BR = self.cellsPositions[i][j][1]

                    text = str(sudokuGame[i, j])
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 5
                    font_color = (255, 0, 0)
                    th = 3

                    text_size = cv.getTextSize(text, font, font_scale, th)[0]
                    pos = ((UL[0] + BR[0]) // 2 - text_size[0] // 2, (UL[1] + BR[1]) // 2 + text_size[1] // 2)

                    copie = cv.putText(copie, text, pos, font, font_scale, font_color, th)
        return copie

    def getAugmentedImage(self, sudokuGame, whereToWrite):
        height, width = self.originalImage.shape[:2]
        img = self.drawResult(sudokuGame, whereToWrite)
        mask = np.ones_like(img, dtype=np.uint8) * 255
        mask = cv.warpPerspective(mask, self.inverse_M, (width, height))
        final = cv.warpPerspective(img, self.inverse_M, (width, height))
        final = final[0:self.image.shape[0], 0:self.image.shape[1], :]
        mask = 255 - mask[0:self.image.shape[0], 0:self.image.shape[1], :]
        result = np.zeros_like(self.image)
        w, h, _ = self.image.shape
        for i in range(w):
            for j in range(h):
                if mask[i, j, 0] == 255 and mask[i, j, 1] == 255 and mask[i, j, 2] == 255:
                    result[i,j] = self.image[i,j]
                else:
                    result[i,j] = final[i,j]

        return result