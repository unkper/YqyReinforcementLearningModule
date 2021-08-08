import math


class Vector3():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # 置为零向量
    def zero(self):
        self.x = 0
        self.y = 0
        self.z = 0

    # "=" 赋值不支持重载

    # 重载 "==" 操作符
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    # 重载 "!=" 操作符
    def __ne__(self, other):
        return self.x != other.x or self.y != other.y or self.z != other.z

    # 重载一元 "-" 操作符
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    # 重载 "+" 操作符
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    # 重载 "-" 操作符
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    # 重载 "*" 操作符
    def __mul__(self, a):
        return Vector3(self.x * a, self.y * a, self.z * a)

    # 重载 "/" 操作符
    def __div__(self, a):
        if (a != 0):
            return Vector3(self.x / a, self.y / a, self.z / a)
        else:
            return None

    # 重载 "+=" 操作符
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    # 重载 "-=" 操作符
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    # 重载 "*=" 操作符
    def __imul__(self, a):
        self.x *= a
        self.y *= a
        self.z *= a
        return self

    # 重载 "/=" 操作符
    def __idiv__(self, a):
        if (a != 0):
            self.x /= a
            self.y /= a
            self.z /= a
        return self

    # 向量标准化
    def normalize(self):
        magSq = self.x * self.x + self.y * self.y + self.z * self.z
        if (magSq > 0):
            oneOverMag = 1.0 / math.sqrt(magSq)
            self.x *= oneOverMag
            self.y *= oneOverMag
            self.z *= oneOverMag

    # 向量求模
    def vectorMag(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    # 向量显示
    def toString(self):
        print("{x:" + str(self.x) + ",y:" + str(self.y) + ",z:" + str(self.z) + "}")


# 向量点乘
def dotProduct(va, vb):
    return va.x * vb.x + va.y * vb.y + va.z * vb.z


# 向量叉乘
def crossProduct(va, vb):
    x = va.y * vb.z - va.z * vb.y
    y = va.z * vb.x - va.x * vb.z
    z = va.x * vb.y - va.y * vb.x
    return Vector3(x, y, z)


# 计算两点间的距离
def distance(va, vb):
    dx = va.x - vb.x
    dy = va.y - vb.y
    dz = va.z - vb.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)