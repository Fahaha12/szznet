

class VIEWS:
    L_CC = "L-CC"
    R_CC = "R-CC"
    L_MLO = "L-MLO"
    R_MLO = "R-MLO"

    LIST = [L_CC, R_CC, L_MLO, R_MLO]


class TWOVIEWS:
    CC = "CC"
    MLO = "MLO"

    LIST = [CC, MLO]


class VIEWANGLES:
    CC = "CC"
    MLO = "MLO"

    LIST = [CC, MLO]


INPUT_SIZE_DICT = {
    VIEWS.L_CC: (2677, 1942),
    VIEWS.R_CC: (2677, 1942),
    VIEWS.L_MLO: (2974, 1748),
    VIEWS.R_MLO: (2974, 1748),
}


MOVE_DIRECTION = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
