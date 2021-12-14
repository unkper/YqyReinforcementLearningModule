


ColorBlue = (30, 144, 255)
ColorGreen = (144, 238, 144)
ColorRed = (255, 91, 71)
ColorYellow = (255, 215, 0)
ColorPurple = (128, 0, 128)
ColorOrange = (255, 127, 0)
ColorPink = (255, 110 , 180)
ColorWall = (205, 133, 63)
ColorWhite = (255, 255, 255)


def exit_type_to_color(exit_type):
    dic = {
        3: ColorRed,
        4: ColorGreen,
        5: ColorPurple,
        6: ColorYellow,
        7: ColorOrange,
        8: ColorPink,
    }
    return dic[exit_type]