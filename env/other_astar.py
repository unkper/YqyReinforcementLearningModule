# coding=utf-8
import copy

chess_size = 20
FILE_STORAGE_WAY = '123.txt'  # 随机数据文件存放路径

Start_x = 0
Start_y = 0
End_x = 0
End_y = 0

OpenListPosition = 0  # 指针标记open列表数组当前存放元素个数（position总指向最后一个元素的后一位置）
CloseListPosition = 0  # 指针标记Close列表数组当前存放元素个数（position总指向最后一个元素的后一位置）

OpenList = []  # open列表(定义成数组形式)
CloseList = []  # close列表

chessboard = []


def openListIncraseSort():
    global OpenListPosition
    global OpenList
    flag = 0
    i = 0
    # for (int i = 0; i < OpenListPosition; i++)//冒泡法由大到小排序
    while i < OpenListPosition:
        # for (int j = 0; j < OpenListPosition - i; j++)
        j = 0
        while j < OpenListPosition - 1:  ##############- 1
            if OpenList[j][2] < OpenList[j + 1][2]:  # 若OpenList[j].f_value < OpenList[j + 1].f_value，交换，从大到小排列

                ChessUnit_x = OpenList[j][0]
                ChessUnit_y = OpenList[j][1]
                f_value = OpenList[j][2]

                OpenList[j][0] = OpenList[j + 1][0]
                OpenList[j][1] = OpenList[j + 1][1]
                OpenList[j][2] = OpenList[j + 1][2]

                OpenList[j + 1][0] = ChessUnit_x
                OpenList[j + 1][1] = ChessUnit_y
                OpenList[j + 1][2] = f_value

                flag = 1  # 将交换标志位置1
            j += 1  # 循环变量增加1
        # if
        # while2
        if (0 == flag):
            break  # 内层循环一次都没交换，说明已经排好序
        flag = 0  # 将标志位重新清0
        i += 1  # 循环变量增加1
    print('===调试用===openListIncraseSort()')  # 调试用，输出openListIncraseSort
    print('===调试用===按f_value排序后的OpenList =')
    print(OpenList)


def dealCurrentNeibors(CurrentUnit):  # 判断当前结点周围8个点状态，计算g、h、f值，将周围每个点的父指针指向当前结点
    global OpenListPosition
    global CloseListPosition
    global OpenList
    global CloseList
    global chessboard
    OpenListFlag = 0
    CloseListFlag = 0

    ObstacleEast = 0  # 标识当前点东边是否有障碍物
    ObstacleSouth = 0  # 标识当前点南边是否有障碍物
    ObstacleWest = 0  # 标识当前点西边是否有障碍物
    ObstacleNorth = 0  # 标识当前点北边是否有障碍物

    # CloseListUnit EastUnit, SourthEastUnit, SourthUnit, SourthWestUnit, WestUnit, NorthWestUnit, NorthUnit, NorthEastUnit;//东，东南，南，西南，西，西北，北，东北
    CurrentNeibors = []  # 列表，按顺序存放当前结点的东，东南，南，西南，西，西北，北，东北方向的邻结点
    OpenlistTemp = []  # 列表，存放openlist【x,y,f_value】中的【x,y】
    for temp in OpenList:
        OpenlistTemp.append([temp[0], temp[1]])
    print('===调试用===dealCurrentNeibors():OpenlistTemp=')  # 调试用，输出OpenlistTemp
    print(OpenlistTemp)  # 调试用，输出OpenlistTemp
    # 东方y+1
    item = [CurrentUnit[0], CurrentUnit[1] + 1]
    CurrentNeibors.append(item)

    # 东南方x+1,y+1
    item = [CurrentUnit[0] + 1, CurrentUnit[1] + 1]
    CurrentNeibors.append(item)

    # 南方x+1
    item = [CurrentUnit[0] + 1, CurrentUnit[1]]
    CurrentNeibors.append(item)

    # 西南方x+1,y-1
    item = [CurrentUnit[0] + 1, CurrentUnit[1] - 1]
    CurrentNeibors.append(item)

    # 西方y-1
    item = [CurrentUnit[0], CurrentUnit[1] - 1]
    CurrentNeibors.append(item)

    # 西北方x-1,y-1
    item = [CurrentUnit[0] - 1, CurrentUnit[1] - 1]
    CurrentNeibors.append(item)

    # 北方x-1
    item = [CurrentUnit[0] - 1, CurrentUnit[1]]
    CurrentNeibors.append(item)

    # 东北方x-1,y+1
    item = [CurrentUnit[0] - 1, CurrentUnit[1] + 1]
    CurrentNeibors.append(item)
    print('===调试用===CurrentUnit=(%d,%d)' % (CurrentUnit[0], CurrentUnit[1]))  # 调试用
    print(chessboard[6][6][4])
    print(chessboard[6][8][4])

    print('===调试用===dealCurrentNeibors():CurrentNeibors=')  # 调试用
    print(CurrentNeibors)  # 调试用
    # 对当前方格东、南、西、北四个方向的临近方格依次检测
    i = 0
    while i < 8:
        # 当前中间结点到邻近结点的距离。约定：东南西北四个方向距离为10，四个斜方向距离为14
        if i % 2 == 0:
            currenToNeibor = 10
        else:
            currenToNeibor = 14
        print('===调试用===dealCurrentNeibors():currenToNeibor=[%d]' % currenToNeibor)  # 调试用
        print(
            '===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (CurrentUnit[0], CurrentUnit[1]))
        print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
              chessboard[CurrentUnit[0]][CurrentUnit[1]][1])
        # 超出边界点
        if CurrentNeibors[i][0] < 0 or CurrentNeibors[i][0] > 19 or CurrentNeibors[i][1] < 0 or CurrentNeibors[i][
            1] > 19:
            print('===调试用===dealCurrentNeibors():超出边界点！！！(%d,%d)' % (CurrentNeibors[i][0], CurrentNeibors[i][1]))
        # continue #结束判断

        # print('===调试用===dealCurrentNeibors():currenToNeibor=[%d],i=[%d]'%(currenToNeibor,i))#调试用
        # print('===调试用===dealCurrentNeibors():CurrentNeibors[%d][0]=[%d]'%(i,CurrentNeibors[i][0]))#调试用
        # print('===调试用===dealCurrentNeibors():CurrentNeibors[%d][1]=[%d]'%(i,CurrentNeibors[i][1]))#调试用
        # print('===调试用===dealCurrentNeibors():chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]=')
        # print(chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4])

        # 终点
        elif 'e' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:  # flag
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
            print('===调试用===dealCurrentNeibors():找到终点！！！(%d,%d)' % (CurrentNeibors[i][0], CurrentNeibors[i][1]))  # 调试用
            return 1

        # 障碍物点
        elif 'b' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:  # flags
            print(
                '===调试用===dealCurrentNeibors():找到障碍物点！！！(%d,%d)' % (CurrentNeibors[i][0], CurrentNeibors[i][1]))  # 调试用
            if i == 0:
                ObstacleEast = 1  # 标识当前点东边有障碍物
                print('===调试用===dealCurrentNeibors():ObstacleEast = %d' % ObstacleEast)
            elif i == 2:
                ObstacleSouth = 1  # 标识当前点南边有障碍物
                print('===调试用===dealCurrentNeibors():ObstacleSouth = %d' % ObstacleSouth)
            elif i == 4:
                ObstacleWest = 1  # 标识当前点西边有障碍物
                print('===调试用===dealCurrentNeibors():ObstacleWest = %d' % ObstacleWest)
            elif i == 6:
                ObstacleNorth = 1  # 标识当前点北边有障碍物
                print('===调试用===dealCurrentNeibors():ObstacleNorth = %d' % ObstacleNorth)

        # 将该临近点与closelist中的点逐个比较
        elif CurrentNeibors[i] in CloseList:  # python牛逼！
            print('===调试用===dealCurrentNeibors():CurrentNeibors[%d]是closelist中的点！！！(%d,%d)' % (
            i, CurrentNeibors[i][0], CurrentNeibors[i][1]))
            print('===调试用===dealCurrentNeibors():CloseList=')
            print(CloseList)
        # continue#结束判断

        # 将该临近点与openlist中的点逐个比较
        elif CurrentNeibors[i] in OpenlistTemp:  # python牛逼！
            print('===调试用===dealCurrentNeibors():CurrentNeibors[i] in OpenlistTemp：')  # 调试用
            print('===调试用===dealCurrentNeibors():CurrentNeibors[i]=')
            print(CurrentNeibors[i])
            print('===调试用===dealCurrentNeibors():OpenlistTemp=')
            print(OpenlistTemp)
            # 若该临近点的g值大于从起点经由当前结点到该临近结点的g值，将该临近结点的父指针指向当前结点，并更改该临近结点的g值，f值
            if chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] > chessboard[CurrentUnit[0]][CurrentUnit[1]][
                1] + currenToNeibor:
                # 将该临近结点的父指针指向当前结点
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
                print('===调试用===dealCurrentNeibors():修改(%d,%d)结点的父节点（%d,%d）' % (
                CurrentNeibors[i][0], CurrentNeibors[i][1], CurrentUnit[0], CurrentUnit[1]))
                # 更改该临近结点的g值，f值
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] = chessboard[CurrentUnit[0]][CurrentUnit[1]][
                                                                                1] + currenToNeibor  # g_value+currenToNeibor
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3] = \
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] + \
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2]  # [1]g[2]h[3]f
                counter = 0
                while len(OpenList) - counter > 0:  # 大bug，漏掉了此处，导致openlist中的f_value值没有被更新！！！注意修改list值不能用for
                    if CurrentNeibors[i][0] == OpenList[counter][0] and CurrentNeibors[i][1] == OpenList[counter][
                        1]:  # 找到openlist中要修改f_value的位置
                        OpenList[counter][2] = chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                            3]  # 更新openlist中的f值
                        break
                    counter += 1
        # 可作为通路点
        elif '#' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:
            # 将邻居结点的指针指向当前结点
            print('===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (
            CurrentUnit[0], CurrentUnit[1]))
            print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
                  chessboard[CurrentUnit[0]][CurrentUnit[1]][1])

            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
            print('===调试用===dealCurrentNeibors():(%d,%d)结点的父节点为（%d,%d）' % (
            CurrentNeibors[i][0], CurrentNeibors[i][1], CurrentUnit[0], CurrentUnit[1]))

            # print('===调试用===dealCurrentNeibors():chessboard=')
            # print(chessboard)

            # 计算该临近结点的g值，h值（曼哈顿距离），f值
            # temptest = chessboard[CurrentUnit[0]][CurrentUnit[1]][1][:] + currenToNeibor
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] = chessboard[CurrentUnit[0]][CurrentUnit[1]][
                                                                            1] + currenToNeibor  # 一条诡异的程序
            print(type(chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1]))
            print(type(chessboard[CurrentUnit[0]][CurrentUnit[1]][1]))
            print(type(currenToNeibor))

            print('===调试用===dealCurrentNeibors():CurrentNeibors[i][0]=[%d] CurrentNeibors[i][1]=[%d]' % (
            CurrentNeibors[i][0], CurrentNeibors[i][1]))
            print('===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (
            CurrentUnit[0], CurrentUnit[1]))
            print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
                  chessboard[CurrentUnit[0]][CurrentUnit[1]][1])
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2] = 10 * (
                        abs(CurrentNeibors[i][0] - End_x) + abs(CurrentNeibors[i][1] - End_y))
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3] = \
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] + \
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2]
            print('===调试用===dealCurrentNeibors():g_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                1])
            print('===调试用===dealCurrentNeibors():h_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                2])
            print('===调试用===dealCurrentNeibors():f_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                3])

            print('===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (
            CurrentUnit[0], CurrentUnit[1]))
            print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
                  chessboard[CurrentUnit[0]][CurrentUnit[1]][1])

            # 将该临近点存入openlist中
            temp = [CurrentNeibors[i][0], CurrentNeibors[i][1],
                    chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3]]
            OpenList.append(temp)
            print('===调试用===dealCurrentNeibors():OpenList.append(temp),temp=')
            print(temp)
            print('===调试用===dealCurrentNeibors():OpenList.append(temp),OpenList')
            print(OpenList)
            OpenListPosition += 1
            print('===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (
            CurrentUnit[0], CurrentUnit[1]))
            print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
                  chessboard[CurrentUnit[0]][CurrentUnit[1]][1])
        # print('===调试用===OpenList.append(temp),OpenListPosition = [%d]',%OpenListPosition)
        i += 2  # 循环变量增加2

    # 对当前方格东南、西南、西北、东北四个方向的临近方格依次检测
    i = 1
    while i < 8:
        # i+=2#循环变量增加2
        # 当前中间结点到邻近结点的距离。约定：东南西北四个方向距离为10，四个斜方向距离为14
        if i % 2 == 0:
            currenToNeibor = 10
        else:
            currenToNeibor = 14
        print('===调试用===dealCurrentNeibors():ObstacleEast = %d' % ObstacleEast)
        print('===调试用===dealCurrentNeibors():ObstacleSouth = %d' % ObstacleSouth)
        print('===调试用===dealCurrentNeibors():ObstacleWest = %d' % ObstacleWest)
        print('===调试用===dealCurrentNeibors():ObstacleNorth = %d' % ObstacleNorth)
        # print('===调试用===i=[%d]'%i)#调试用

        print('===调试用===dealCurrentNeibors():currenToNeibor=[%d]' % currenToNeibor)  # 调试用

        if 1 == ObstacleEast and (1 == i or 7 == i):  # 若东方格是障碍物，则东南、东北都不能通行
            print('===调试用===i=[%d]' % i)  # 调试用
            i += 2
            continue

        if 1 == ObstacleSouth and (1 == i or 3 == i):  # 若南方格是障碍物，则东南、西南都不能通行
            print('===调试用===i=[%d]' % i)  # 调试用
            i += 2
            continue

        if 1 == ObstacleWest and (3 == i or 5 == i):  # 若西方格是障碍物，则西南、西北都不能通行
            print('===调试用===i=[%d]' % i)  # 调试用
            i += 2
            continue

        if 1 == ObstacleNorth and (5 == i or 7 == i):  # 若北方格是障碍物，则西北、东北都不能通行
            print('===调试用===i=[%d]' % i)  # 调试用
            i += 2
            continue

        # 超出边界点
        if CurrentNeibors[i][0] < 0 or CurrentNeibors[i][0] > 19 or CurrentNeibors[i][1] < 0 or CurrentNeibors[i][
            1] > 19:
            print('===调试用===dealCurrentNeibors():超出边界点！！！(%d,%d)' % (CurrentNeibors[i][0], CurrentNeibors[i][1]))
        # continue #结束判断

        # 终点
        elif 'e' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:  # flag
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
            print('===调试用===dealCurrentNeibors():找到终点！！！[%d][%d]' % (
            CurrentNeibors[i][0], CurrentNeibors[i][1]))  # 调试用
            return 1

        # 障碍物点
        elif 'b' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:  # flags
            print(
                '===调试用===dealCurrentNeibors():找到障碍物点！！！[%d][%d]' % (CurrentNeibors[i][0], CurrentNeibors[i][1]))  # 调试用


        # 将该临近点与closelist中的点逐个比较
        elif CurrentNeibors[i] in CloseList:  # python牛逼！
            print('===调试用===dealCurrentNeibors():CurrentNeibors[%d]是closelist中的点！！！(%d,%d)' % (
            i, CurrentNeibors[i][0], CurrentNeibors[i][1]))
        # continue#结束判断

        # 将该临近点与openlist中的点逐个比较
        elif CurrentNeibors[i] in OpenlistTemp:  # python牛逼！
            print('===调试用===dealCurrentNeibors():CurrentNeibors[i] in OpenlistTemp：')  # 调试用
            print('===调试用===dealCurrentNeibors():CurrentNeibors[i]=')
            print(CurrentNeibors[i])
            print('===调试用===dealCurrentNeibors():OpenlistTemp=')
            print(OpenlistTemp)
            # 若该临近点的g值大于从起点经由当前结点到该临近结点的g值，将该临近结点的父指针指向当前结点，并更改该临近结点的g值，f值
            if chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] > chessboard[CurrentUnit[0]][CurrentUnit[1]][
                1] + currenToNeibor:
                # 将该临近结点的父指针指向当前结点
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
                print('===调试用===dealCurrentNeibors():修改(%d,%d)结点的父节点（%d,%d）' % (
                CurrentNeibors[i][0], CurrentNeibors[i][1], CurrentUnit[0], CurrentUnit[1]))

                # 更改该临近结点的g值，f值
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] = chessboard[CurrentUnit[0]][CurrentUnit[1]][
                                                                                1] + currenToNeibor  # g_value+currenToNeibor
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3] = \
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] + \
                chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2]  # [1]g[2]h[3]f
                counter = 0
                while len(OpenList) - counter > 0:  # 大bug，漏掉了此处，导致openlist中的f_value值没有被更新！！！注意修改list值不能用for
                    if CurrentNeibors[i][0] == OpenList[counter][0] and CurrentNeibors[i][1] == OpenList[counter][
                        1]:  # 找到openlist中要修改f_value的位置
                        OpenList[counter][2] = chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                            3]  # 更新openlist中的f值
                        break
                    counter += 1
        # 可作为通路点
        elif '#' == chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][4]:  # flag
            # 将邻居结点的指针指向当前结点
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][0] = CurrentUnit[0]  # [0][0]parent.x
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][0][1] = CurrentUnit[1]  # [0][1]parent.y
            print('===调试用===dealCurrentNeibors():(%d,%d)结点的父节点为（%d,%d）' % (
            CurrentNeibors[i][0], CurrentNeibors[i][1], CurrentUnit[0], CurrentUnit[1]))

            # 计算该临近结点的g值，h值（曼哈顿距离），f值
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] = chessboard[CurrentUnit[0]][CurrentUnit[1]][
                                                                            1] + currenToNeibor
            print('===调试用===dealCurrentNeibors():CurrentUnit[0]=[%d] CurrentUnit[1]=[%d]' % (
            CurrentUnit[0], CurrentUnit[1]))
            print('===调试用===dealCurrentNeibors():chessboard[CurrentUnit[0]][CurrentUnit[1]][1]=[%d]' %
                  chessboard[CurrentUnit[0]][CurrentUnit[1]][1])
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2] = 10 * (
                        abs(CurrentNeibors[i][0] - End_x) + abs(CurrentNeibors[i][1] - End_y))
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3] = \
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][1] + \
            chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][2]
            print('===调试用===dealCurrentNeibors():g_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                1])
            print('===调试用===dealCurrentNeibors():h_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                2])
            print('===调试用===dealCurrentNeibors():f_value=[%d]' % chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][
                3])

            # 将该临近点存入openlist中
            temp = [CurrentNeibors[i][0], CurrentNeibors[i][1],
                    chessboard[CurrentNeibors[i][0]][CurrentNeibors[i][1]][3]]
            OpenList.append(temp)
            print('===调试用===dealCurrentNeibors():OpenList.append(temp),temp=')
            print(temp)
            print('===调试用===dealCurrentNeibors():OpenList.append(temp),OpenList')
            print(OpenList)
            OpenListPosition += 1
        #	print('===调试用===OpenList.append(temp),OpenListPosition = [%d]',%OpenListPosition)
        i += 2  # 循环变量增加2
    return 0


def FileReadMatrix():  # 将文件中的数据读回至chess_size*chess_size矩阵
    global Start_x
    global Start_y
    global End_x
    global End_y

    itemLine = []
    fileLine = []
    chessboardTemp = []
    ifile = open(FILE_STORAGE_WAY, 'r')  # 作为只读文件打开
    fileLine = ifile.readline()
    while len(fileLine) > 0:  # 将迷宫从txt文件中读入到chessboardTemp列表中
        print(fileLine)
        print(fileLine[0:20])
        chessboardTemp.append(fileLine[0:20])
        fileLine = ifile.readline()
    print(chessboardTemp)
    print(chessboardTemp[2][2])
    i = 0
    while i < 20:
        j = 0
        while j < 20:
            # 将棋盘读入到flag中
            if 's' == chessboardTemp[i][j]:  # 检测并记录起始点坐标
                print('起始点坐标！！！[%d,%d]' % (i, j))
                Start_x = i
                Start_y = j
            if 'e' == chessboardTemp[i][j]:  # 检测并记录终止点坐标
                print('终止点坐标！！！[%d,%d]' % (i, j))
                End_x = i
                End_y = j
            if 'b' == chessboardTemp[i][j]:  # 检测并记录终止点坐标
                print('障碍点坐标！！！[%d,%d]' % (i, j))

            item = [[0, 0], 0, 0, 0, chessboardTemp[i][j]]  # 结构：A = [[parent.x,parent.y],g_value,h_value,f_value,flag]
            # print(item[4])
            itemLine.append(copy.deepcopy(item))  # 结构：B = [A,A,A,A,A....,A]一行20个,一定要传切片[:]！！！
            print(itemLine[j][4])
            j += 1
        print(fileLine)
        chessboard.append(copy.deepcopy(itemLine))  # 结构：[B,B,B,B,...,B] 20列，一定要传切片！！！
        itemLine = []  # 超级大bug：将itemLine清空（查了一晚上！！！！）
        fileLine = ifile.readline()  # 读取下一行数据
        i += 1
    print(chessboard[2][2][4])


# print(chessboard[6][8][4])
# print(chessboard)
def printPath():
    PathUnit = []  # 逆序存放单路径结点（终点->起点）
    PathUnitList = []  # 逆序存放所有路径结点（终点->起点）
    # 获取终点的坐标
    PathUnit = [End_x, End_y]

    # cout << "(" << PathUnit.ChessUnit_x << "," << PathUnit.ChessUnit_y << ")" << endl;//输出终点坐标
    print('(%d,%d)' % (PathUnit[0], PathUnit[1]))
    # 记录从end起第一个最佳路径结点
    PathUnit = [chessboard[End_x][End_y][0][0], chessboard[End_x][End_y][0][1]]
    i = 0  # 循环变量
    while not (PathUnit[0] == Start_x and PathUnit[1] == Start_y):  # 记录从终点到起点之间的最佳路径
        PathUnitList.append(copy.deepcopy(PathUnit))  # 重大bug！！！必须传拷贝，不然循环插入到会修改上一次PathUnit的值
        chessboard[PathUnitList[i][0]][PathUnitList[i][1]][4] = '*'  # 将最佳路径点用"*"表示
        # cout << "(" << PathUnitList[i].ChessUnit_x << "," << PathUnitList[i].ChessUnit_y << ")" << endl;//输出路径结点坐标
        # 输出路径结点坐标
        print('(%d,%d)' % (PathUnit[0], PathUnit[1]))
        # 获取当前结点的父节点坐标
        PathUnit[0] = chessboard[PathUnitList[i][0]][PathUnitList[i][1]][0][0]
        PathUnit[1] = chessboard[PathUnitList[i][0]][PathUnitList[i][1]][0][1]
        i += 1
    # cout << "(" << Start_x << "," << Start_y << ")" << endl;//输出终点坐标
    print('(%d,%d)' % (Start_x, Start_y))  # 输出终点坐标
    '''
    for (int i = 0; i < chess_size; i++)
    {
        for (int j = 0; j < chess_size; j++)
        {
            cout << chessboard[i][j].flag;
        }
        cout << endl;
    }
    '''

    temp = ''
    i = 0
    while i < 20:
        j = 0
        while j < 20:
            temp += chessboard[i][j][4]
            j += 1
        print(temp)
        temp = ''  # 清空行输出变量
        i += 1


def main():
    global OpenListPosition
    global CloseListPosition
    global OpenList
    global CloseList
    global Start_x
    global Start_y

    FileReadMatrix()

    item = [Start_x, Start_y, 0]  # 将起始点x坐标，y坐标，f值封装到列表item中
    print(item)
    OpenList.append(item)  # 将item插入到Openlist中

    OpenListPosition += 1  # 将起始结点存入openlist，position标记为1（position总指向最后一个元素的后一位置）

    while OpenListPosition > 0:  # 若openlist不为空
        openListIncraseSort()  # openlist列表按f值大小降序排序（将最小f值点放在最后，这样只需将position减1就代表移出该点）
        # 将openlist中f值最小的点移入closelist中
        item = [OpenList[OpenListPosition - 1][0], OpenList[OpenListPosition - 1][1]]
        CloseList.append(item)
        # print(CloseList)#调试用
        # openlist移出f值最小元素，清除原位置该元素信息
        OpenList.pop()  # 移除openlist中最后的那个元素
        print('===调试用===main():移除最小元素之后的openlist=')
        print(OpenList)
        OpenListPosition -= 1  # 将OpenListPosition减1，表示从openlist中移出最后一点，即f值最小的点
        # print(OpenListPosition)
        CloseListPosition += 1  # 将ClosePosition加1，记录closelist中增加一个元素

        # 将f最小结点信息插入到CurrentUnit中
        CurrentUnit = [CloseList[CloseListPosition - 1][0], CloseList[CloseListPosition - 1][1]]
        print('===调试用===main():CloseListPosition=[%d]' % CloseListPosition)
        print('===调试用===main():CloseList=')
        print('===调试用===main():CloseList=')
        print(CloseList)
        # 判断当前结点周围8个点状态，计算g、h、f值，将周围每个点的父指针指向当前结点
        if dealCurrentNeibors(CurrentUnit) == 1:
            print('===调试用===main():找到终点，跳出循环')
            print('===调试用===main():找到终点，跳出循环')
            print('===调试用===main():找到终点，跳出循环')
            break
    # dealCurrentNeibors(CurrentUnit)
    print('===调试用===OpenListPosition=[%d]' % OpenListPosition)
    # while
    printPath()


# return 0

main()