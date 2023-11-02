from imports import np
from imports import plp

class Sudoku:
    def __init__(self, game):
        self.game = np.zeros((9, 9), dtype=np.uint8)

        for i in range(9):
            for j in range(9):
                if game[i][j] != 'o':
                    self.game[i, j] = int(game[i][j])


    @classmethod
    def solveSudoku(self, game):
        problem = plp.LpProblem("SolveSudoku")
        problem.setObjective(plp.lpSum(0))
        rows = range(0, 9)
        cols = range(0, 9)
        grids = [[(3 * i + k, 3 * j + l) for k in range(3) for l in range(3)] for i in range(3) for j in range(3)]
        values = range(1, 10)
        choices  = plp.LpVariable.dicts("Choice", (rows, cols, values), cat='Binary')

        # one value in each square
        for r in rows:
            for c in cols:
                problem += plp.lpSum([choices[r][c][v] for v in values]) == 1

        # each number only occur once in each row
        for v in values:
            for r in rows:
                problem += plp.lpSum([choices[r][c][v] for c in cols]) == 1

        # each number only occur once in each column
        for v in values:
            for c in cols:
                problem += plp.lpSum([choices[r][c][v] for r in rows]) == 1

        # each number only occur once in each box
        for v in values:
            for g in grids:
                problem += plp.lpSum([choices[r][c][v] for (r, c) in g]) == 1

        for i in range(0, 9):
            for j in range(0, 9):
                if game[i, j] != 0:
                    problem += choices[i][j][game[i, j]] == 1

        problem.solve()
        solution = game.copy()

        for r in rows:
            for c in cols:
                for v in values:
                    if plp.value(choices[r][c][v]) == 1:
                        solution[r, c] = v

        return solution