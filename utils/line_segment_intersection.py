"""
    #define max(a,b) ((a)>(b)?(a):(b))
    #define min(a,b) ((a)<(b)?(a):(b))
    typedef struct{
        double x;
        double y;
    }Point;
    typedef struct{
        Point A,B;
    }Line;

"""


def _max(a, b):
    return a if a > b else b
    # if a > b:
    #     return a
    # else:
    #     return b


def _min(a, b):
    if a < b:
        return a
    else:
        return b


#  线段L与C的位置关系
# 返回值： > 0 即C在L的顺时针方向（下方）； < 0即C在L的逆时针方向（上方）；=0同线
# l(A,B)  a(x,y)  b(x,y)  c(x,y)
def point_line_location(c, l):
    # AB_x, AB_y  # 矢量AB
    # AC_x, AC_y  # 矢量AC
    ab_x = l.A.x - l.B.x
    ab_y = l.B.y - l.B.y  # 矢量A->B
    ac_x = l.A.x - c.x
    ac_y = l.A.y - c.y  # 矢量 A->C
    # return ab_x * ac_y - ab_y * ab_x #  矢量AB、AC叉乘
    return ab_x * ac_y - ab_y * ac_x  # 矢量AB、AC叉乘


def intersection(l1, l2):  # l1和l2是否相交
    # 快速排斥
    if max(l1.A.x, l1.B.x) < min(l2.A.x, l2.B.x):
        return 0
    if max(l1.A.y, l1.B.y) < min(l2.A.y, l2.A.y):
        return 0
    if max(l2.A.x, l2.B.x) < min(l1.A.x, l1.A.x):
        return 0
    if max(l2.A.y, l2.B.y) < min(l1.A.y, l1.A.y):
        return 0

    # 跨立实验
    if point_line_location(l1.A, l2) * point_line_location(l1.B, l2) > 0:  # l1的两端点都在l2的同侧
        return 0
    if point_line_location(l2.A, l1) * point_line_location(l2.B, l1) > 0:  # l2的两端点都在l1的同侧
        return 0

    return 1  # 相交
