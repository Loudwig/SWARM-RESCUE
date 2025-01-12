"""
This file was generated by the tool 'image_to_map.py' in the directory tools.
This tool permits to create this kind of file by providing it an image of the
map we want to create.
"""
import sys
from pathlib import Path

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (1700, 1100)
# Dimension factor : 1.0
def add_boxes(playground):
    pass


def add_walls(playground):
    # horizontal wall 0
    wall = NormalWall(pos_start=(445, 546),
                      pos_end=(476, 546))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 1
    wall = NormalWall(pos_start=(438, 546),
                      pos_end=(438, 422))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 2
    wall = NormalWall(pos_start=(407, 424),
                      pos_end=(440, 424))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 3
    wall = NormalWall(pos_start=(473, 546),
                      pos_end=(473, 412))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 4
    wall = NormalWall(pos_start=(471, 415),
                      pos_end=(741, 415))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 5
    wall = NormalWall(pos_start=(739, 416),
                      pos_end=(739, 393))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(440, 544),
                      pos_end=(440, 419))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 7
    wall = NormalWall(pos_start=(405, 422),
                      pos_end=(442, 422))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 8
    wall = NormalWall(pos_start=(475, 544),
                      pos_end=(475, 415))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 9
    wall = NormalWall(pos_start=(473, 417),
                      pos_end=(743, 417))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 10
    wall = NormalWall(pos_start=(741, 419),
                      pos_end=(741, 390))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 11
    wall = NormalWall(pos_start=(329, 428),
                      pos_end=(353, 462))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 12
    wall = NormalWall(pos_start=(346, 463),
                      pos_end=(415, 420))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 13
    wall = NormalWall(pos_start=(347, 459),
                      pos_end=(413, 418))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 14
    wall = NormalWall(pos_start=(333, 429),
                      pos_end=(353, 457))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 15
    wall = NormalWall(pos_start=(-730, 444),
                      pos_end=(-730, 190))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 16
    wall = NormalWall(pos_start=(-730, 442),
                      pos_end=(-654, 442))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 17
    wall = NormalWall(pos_start=(-657, 444),
                      pos_end=(-657, 411))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 18
    wall = NormalWall(pos_start=(-571, 443),
                      pos_end=(-571, 408))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 19
    wall = NormalWall(pos_start=(-571, 441),
                      pos_end=(130, 441))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 20
    wall = NormalWall(pos_start=(123, 443),
                      pos_end=(157, 420))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 21
    wall = NormalWall(pos_start=(-727, 440),
                      pos_end=(-657, 440))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 22
    wall = NormalWall(pos_start=(-659, 442),
                      pos_end=(-659, 414))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 23
    wall = NormalWall(pos_start=(-728, 443),
                      pos_end=(-728, 193))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 24
    wall = NormalWall(pos_start=(-568, 439),
                      pos_end=(128, 439))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 25
    wall = NormalWall(pos_start=(121, 442),
                      pos_end=(157, 417))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 26
    wall = NormalWall(pos_start=(-569, 442),
                      pos_end=(-569, 411))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 27
    wall = NormalWall(pos_start=(335, 434),
                      pos_end=(408, 388))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 28
    wall = NormalWall(pos_start=(402, 393),
                      pos_end=(741, 393))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 29
    wall = NormalWall(pos_start=(333, 432),
                      pos_end=(406, 386))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 30
    wall = NormalWall(pos_start=(400, 391),
                      pos_end=(744, 391))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 31
    wall = NormalWall(pos_start=(129, 388),
                      pos_end=(154, 425))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 32
    wall = NormalWall(pos_start=(-699, 414),
                      pos_end=(-656, 414))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 33
    wall = NormalWall(pos_start=(-696, 416),
                      pos_end=(-696, 193))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 34
    wall = NormalWall(pos_start=(-728, 196),
                      pos_end=(-695, 196))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 35
    wall = NormalWall(pos_start=(132, 388),
                      pos_end=(156, 422))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 36
    wall = NormalWall(pos_start=(-570, 413),
                      pos_end=(-530, 413))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 37
    wall = NormalWall(pos_start=(-533, 414),
                      pos_end=(-533, 200))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 38
    wall = NormalWall(pos_start=(-535, 202),
                      pos_end=(-505, 202))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 39
    wall = NormalWall(pos_start=(-697, 412),
                      pos_end=(-655, 412))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 40
    wall = NormalWall(pos_start=(-694, 413),
                      pos_end=(-694, 191))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 41
    wall = NormalWall(pos_start=(-732, 194),
                      pos_end=(-693, 194))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 42
    wall = NormalWall(pos_start=(-573, 411),
                      pos_end=(-532, 411))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 43
    wall = NormalWall(pos_start=(-535, 412),
                      pos_end=(-535, 197))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 44
    wall = NormalWall(pos_start=(-536, 200),
                      pos_end=(-502, 200))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 45
    wall = NormalWall(pos_start=(25, 259),
                      pos_end=(115, 409))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 46
    wall = NormalWall(pos_start=(-217, 263),
                      pos_end=(31, 263))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 47
    wall = NormalWall(pos_start=(-289, 188),
                      pos_end=(-210, 265))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 48
    wall = NormalWall(pos_start=(-309, 211),
                      pos_end=(-284, 187))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 49
    wall = NormalWall(pos_start=(109, 410),
                      pos_end=(137, 391))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 50
    wall = NormalWall(pos_start=(-505, 406),
                      pos_end=(-505, 199))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 51
    wall = NormalWall(pos_start=(-505, 404),
                      pos_end=(-399, 404))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 52
    wall = NormalWall(pos_start=(-402, 406),
                      pos_end=(-402, 306))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 53
    wall = NormalWall(pos_start=(-404, 313),
                      pos_end=(-380, 290))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 54
    wall = NormalWall(pos_start=(-385, 289),
                      pos_end=(-267, 406))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 55
    wall = NormalWall(pos_start=(-273, 401),
                      pos_end=(69, 401))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 56
    wall = NormalWall(pos_start=(-386, 291),
                      pos_end=(-270, 407))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 57
    wall = NormalWall(pos_start=(-277, 403),
                      pos_end=(75, 403))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 58
    wall = NormalWall(pos_start=(-502, 402),
                      pos_end=(-402, 402))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 59
    wall = NormalWall(pos_start=(-404, 404),
                      pos_end=(-404, 304))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 60
    wall = NormalWall(pos_start=(-406, 310),
                      pos_end=(-381, 286))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 61
    wall = NormalWall(pos_start=(110, 406),
                      pos_end=(134, 390))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 62
    wall = NormalWall(pos_start=(-503, 405),
                      pos_end=(-503, 197))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 63
    wall = NormalWall(pos_start=(27, 256),
                      pos_end=(116, 404))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 64
    wall = NormalWall(pos_start=(-216, 261),
                      pos_end=(33, 261))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 65
    wall = NormalWall(pos_start=(-288, 184),
                      pos_end=(-209, 263))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 66
    wall = NormalWall(pos_start=(3, 284),
                      pos_end=(75, 403))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 67
    wall = NormalWall(pos_start=(-231, 288),
                      pos_end=(9, 288))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 68
    wall = NormalWall(pos_start=(-307, 208),
                      pos_end=(-224, 290))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 69
    wall = NormalWall(pos_start=(1, 286),
                      pos_end=(69, 398))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 70
    wall = NormalWall(pos_start=(-233, 290),
                      pos_end=(7, 290))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 71
    wall = NormalWall(pos_start=(-309, 209),
                      pos_end=(-226, 292))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 72
    wall = NormalWall(pos_start=(235, 378),
                      pos_end=(348, 309))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 73
    wall = NormalWall(pos_start=(344, 315),
                      pos_end=(344, 162))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 74
    wall = NormalWall(pos_start=(343, 163),
                      pos_end=(533, 163))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 75
    wall = NormalWall(pos_start=(530, 164),
                      pos_end=(530, 135))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 76
    wall = NormalWall(pos_start=(212, 333),
                      pos_end=(240, 376))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 77
    wall = NormalWall(pos_start=(236, 374),
                      pos_end=(347, 306))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 78
    wall = NormalWall(pos_start=(342, 312),
                      pos_end=(342, 159))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 79
    wall = NormalWall(pos_start=(340, 161),
                      pos_end=(530, 161))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 80
    wall = NormalWall(pos_start=(528, 162),
                      pos_end=(528, 138))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 81
    wall = NormalWall(pos_start=(215, 333),
                      pos_end=(241, 372))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 82
    wall = NormalWall(pos_start=(217, 339),
                      pos_end=(305, 285))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 83
    wall = NormalWall(pos_start=(301, 291),
                      pos_end=(301, -7))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 84
    wall = NormalWall(pos_start=(89, -366),
                      pos_end=(303, 0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 85
    wall = NormalWall(pos_start=(-265, -361),
                      pos_end=(95, -361))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 86
    wall = NormalWall(pos_start=(218, 335),
                      pos_end=(304, 282))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 87
    wall = NormalWall(pos_start=(299, 288),
                      pos_end=(299, -5))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 88
    wall = NormalWall(pos_start=(88, -362),
                      pos_end=(301, 2))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 89
    wall = NormalWall(pos_start=(-262, -359),
                      pos_end=(94, -359))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 90
    wall = NormalWall(pos_start=(459, 303),
                      pos_end=(844, 303))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 91
    wall = NormalWall(pos_start=(462, 304),
                      pos_end=(462, 268))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 92
    wall = NormalWall(pos_start=(460, 307),
                      pos_end=(460, 263))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 93
    wall = NormalWall(pos_start=(460, 305),
                      pos_end=(844, 305))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 94
    wall = NormalWall(pos_start=(459, 268),
                      pos_end=(662, 268))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 95
    wall = NormalWall(pos_start=(659, 270),
                      pos_end=(659, 7))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 96
    wall = NormalWall(pos_start=(703, 268),
                      pos_end=(844, 268))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 97
    wall = NormalWall(pos_start=(705, 270),
                      pos_end=(705, -27))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 98
    wall = NormalWall(pos_start=(456, -25),
                      pos_end=(707, -25))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 99
    wall = NormalWall(pos_start=(703, 272),
                      pos_end=(703, -26))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 100
    wall = NormalWall(pos_start=(461, -23),
                      pos_end=(704, -23))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 101
    wall = NormalWall(pos_start=(703, 270),
                      pos_end=(844, 270))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 102
    wall = NormalWall(pos_start=(458, 266),
                      pos_end=(660, 266))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 103
    wall = NormalWall(pos_start=(657, 267),
                      pos_end=(657, 10))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 104
    wall = NormalWall(pos_start=(-306, 213),
                      pos_end=(-283, 189))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 105
    wall = NormalWall(pos_start=(-379, 156),
                      pos_end=(-349, 127))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 106
    wall = NormalWall(pos_start=(-420, 110),
                      pos_end=(-375, 154))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 107
    wall = NormalWall(pos_start=(-415, 116),
                      pos_end=(-415, -105))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 108
    wall = NormalWall(pos_start=(-511, -103),
                      pos_end=(-413, -103))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 109
    wall = NormalWall(pos_start=(-380, 153),
                      pos_end=(-353, 128))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 110
    wall = NormalWall(pos_start=(-418, 107),
                      pos_end=(-374, 151))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 111
    wall = NormalWall(pos_start=(-413, 113),
                      pos_end=(-413, -108))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 112
    wall = NormalWall(pos_start=(-514, -105),
                      pos_end=(-411, -105))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 113
    wall = NormalWall(pos_start=(339, 138),
                      pos_end=(531, 138))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 114
    wall = NormalWall(pos_start=(342, 140),
                      pos_end=(342, -17))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 115
    wall = NormalWall(pos_start=(116, -399),
                      pos_end=(344, -10))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 116
    wall = NormalWall(pos_start=(-305, -395),
                      pos_end=(122, -395))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 117
    wall = NormalWall(pos_start=(341, 136),
                      pos_end=(532, 136))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 118
    wall = NormalWall(pos_start=(344, 137),
                      pos_end=(344, -18))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 119
    wall = NormalWall(pos_start=(118, -401),
                      pos_end=(347, -11))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 120
    wall = NormalWall(pos_start=(-308, -397),
                      pos_end=(124, -397))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 121
    wall = NormalWall(pos_start=(-386, 100),
                      pos_end=(-354, 132))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 122
    wall = NormalWall(pos_start=(-381, 106),
                      pos_end=(-381, -140))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 123
    wall = NormalWall(pos_start=(-383, -133),
                      pos_end=(-259, -257))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 124
    wall = NormalWall(pos_start=(-264, -251),
                      pos_end=(-264, -364))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 125
    wall = NormalWall(pos_start=(-384, 98),
                      pos_end=(-353, 130))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 126
    wall = NormalWall(pos_start=(-379, 104),
                      pos_end=(-379, -138))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 127
    wall = NormalWall(pos_start=(-381, -131),
                      pos_end=(-257, -255))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 128
    wall = NormalWall(pos_start=(-262, -249),
                      pos_end=(-262, -361))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 129
    wall = NormalWall(pos_start=(-735, 106),
                      pos_end=(-735, -139))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 130
    wall = NormalWall(pos_start=(-735, 104),
                      pos_end=(-704, 104))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 131
    wall = NormalWall(pos_start=(-707, 106),
                      pos_end=(-707, -107))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 132
    wall = NormalWall(pos_start=(-708, -104),
                      pos_end=(-538, -104))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 133
    wall = NormalWall(pos_start=(-538, 105),
                      pos_end=(-538, -107))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 134
    wall = NormalWall(pos_start=(-538, 103),
                      pos_end=(-508, 103))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 135
    wall = NormalWall(pos_start=(-511, 105),
                      pos_end=(-511, -106))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 136
    wall = NormalWall(pos_start=(-732, 102),
                      pos_end=(-707, 102))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 137
    wall = NormalWall(pos_start=(-709, 104),
                      pos_end=(-709, -109))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 138
    wall = NormalWall(pos_start=(-711, -106),
                      pos_end=(-535, -106))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 139
    wall = NormalWall(pos_start=(-733, 105),
                      pos_end=(-733, -136))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 140
    wall = NormalWall(pos_start=(-535, 101),
                      pos_end=(-511, 101))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 141
    wall = NormalWall(pos_start=(-513, 103),
                      pos_end=(-513, -108))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 142
    wall = NormalWall(pos_start=(-536, 104),
                      pos_end=(-536, -109))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 143
    wall = NormalWall(pos_start=(-191, 61),
                      pos_end=(-191, -104))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 144
    wall = NormalWall(pos_start=(-191, 59),
                      pos_end=(93, 59))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 145
    wall = NormalWall(pos_start=(90, 61),
                      pos_end=(90, -100))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 146
    wall = NormalWall(pos_start=(-188, 57),
                      pos_end=(90, 57))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 147
    wall = NormalWall(pos_start=(88, 59),
                      pos_end=(88, -97))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 148
    wall = NormalWall(pos_start=(-189, 60),
                      pos_end=(-189, -100))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 149
    wall = NormalWall(pos_start=(-161, 32),
                      pos_end=(-161, -99))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 150
    wall = NormalWall(pos_start=(-189, -97),
                      pos_end=(-159, -97))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 151
    wall = NormalWall(pos_start=(-161, 30),
                      pos_end=(68, 30))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 152
    wall = NormalWall(pos_start=(65, 32),
                      pos_end=(65, -98))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 153
    wall = NormalWall(pos_start=(64, -97),
                      pos_end=(90, -97))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 154
    wall = NormalWall(pos_start=(-158, 28),
                      pos_end=(65, 28))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 155
    wall = NormalWall(pos_start=(63, 30),
                      pos_end=(63, -101))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 156
    wall = NormalWall(pos_start=(61, -99),
                      pos_end=(93, -99))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 157
    wall = NormalWall(pos_start=(-159, 31),
                      pos_end=(-159, -102))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 158
    wall = NormalWall(pos_start=(-192, -99),
                      pos_end=(-158, -99))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 159
    wall = NormalWall(pos_start=(456, 12),
                      pos_end=(658, 12))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 160
    wall = NormalWall(pos_start=(459, 13),
                      pos_end=(459, -26))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 161
    wall = NormalWall(pos_start=(458, 10),
                      pos_end=(661, 10))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 162
    wall = NormalWall(pos_start=(461, 11),
                      pos_end=(461, -23))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 163
    wall = NormalWall(pos_start=(369, -132),
                      pos_end=(844, -132))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 164
    wall = NormalWall(pos_start=(231, -375),
                      pos_end=(376, -130))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 165
    wall = NormalWall(pos_start=(369, -130),
                      pos_end=(844, -130))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 166
    wall = NormalWall(pos_start=(229, -373),
                      pos_end=(374, -127))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 167
    wall = NormalWall(pos_start=(-734, -134),
                      pos_end=(-434, -134))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 168
    wall = NormalWall(pos_start=(-441, -133),
                      pos_end=(-298, -276))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 169
    wall = NormalWall(pos_start=(-303, -270),
                      pos_end=(-303, -396))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 170
    wall = NormalWall(pos_start=(-737, -136),
                      pos_end=(-436, -136))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 171
    wall = NormalWall(pos_start=(-443, -135),
                      pos_end=(-300, -278))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 172
    wall = NormalWall(pos_start=(-305, -272),
                      pos_end=(-305, -399))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 173
    wall = NormalWall(pos_start=(384, -166),
                      pos_end=(844, -166))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 174
    wall = NormalWall(pos_start=(254, -396),
                      pos_end=(391, -164))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 175
    wall = NormalWall(pos_start=(228, -374),
                      pos_end=(260, -396))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 176
    wall = NormalWall(pos_start=(251, -394),
                      pos_end=(390, -160))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 177
    wall = NormalWall(pos_start=(384, -164),
                      pos_end=(844, -164))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 178
    wall = NormalWall(pos_start=(844, 548),
                      pos_end=(844, -547))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 179
    wall = NormalWall(pos_start=(-674, -256),
                      pos_end=(-674, -442))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 180
    wall = NormalWall(pos_start=(-674, -258),
                      pos_end=(-481, -258))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 181
    wall = NormalWall(pos_start=(-488, -256),
                      pos_end=(-425, -319))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 182
    wall = NormalWall(pos_start=(-430, -313),
                      pos_end=(-430, -546))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 183
    wall = NormalWall(pos_start=(-671, -260),
                      pos_end=(-483, -260))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 184
    wall = NormalWall(pos_start=(-490, -258),
                      pos_end=(-427, -321))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 185
    wall = NormalWall(pos_start=(-432, -315),
                      pos_end=(-432, -544))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 186
    wall = NormalWall(pos_start=(-672, -257),
                      pos_end=(-672, -439))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 187
    wall = NormalWall(pos_start=(-646, -289),
                      pos_end=(-646, -416))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 188
    wall = NormalWall(pos_start=(-646, -291),
                      pos_end=(-497, -291))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 189
    wall = NormalWall(pos_start=(-504, -289),
                      pos_end=(-455, -338))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 190
    wall = NormalWall(pos_start=(-460, -332),
                      pos_end=(-460, -546))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 191
    wall = NormalWall(pos_start=(-643, -293),
                      pos_end=(-499, -293))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 192
    wall = NormalWall(pos_start=(-506, -291),
                      pos_end=(-457, -340))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 193
    wall = NormalWall(pos_start=(-462, -334),
                      pos_end=(-462, -544))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 194
    wall = NormalWall(pos_start=(-644, -290),
                      pos_end=(-644, -413))
    playground.add(wall, wall.wall_coordinates)

    # oblique wall 195
    wall = NormalWall(pos_start=(232, -373),
                      pos_end=(259, -392))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 196
    wall = NormalWall(pos_start=(-645, -411),
                      pos_end=(-552, -411))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 197
    wall = NormalWall(pos_start=(-555, -409),
                      pos_end=(-555, -440))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 198
    wall = NormalWall(pos_start=(-648, -413),
                      pos_end=(-555, -413))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 199
    wall = NormalWall(pos_start=(-557, -411),
                      pos_end=(-557, -437))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 200
    wall = NormalWall(pos_start=(-673, -437),
                      pos_end=(-554, -437))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 201
    wall = NormalWall(pos_start=(-676, -439),
                      pos_end=(-553, -439))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 202
    wall = NormalWall(pos_start=(-847, -544),
                      pos_end=(846, -544))
    playground.add(wall, wall.wall_coordinates)

