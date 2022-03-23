def startwith(start: int, mgraph: list) -> list:
    """
    返回各点关于start的最短距离
    :param start: 出发点的索引，从0开始
    :param mgraph: mgraph 设置好的邻接矩阵，二维numpy数组
    :return: 一个列表，各元素值是从start到对应下标的点的最短距离
    """
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]
    dis = mgraph[start]
    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]:
                idx = i
        nopass.remove(idx)
        passed.append(idx)
        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + mgraph[idx][i]
    res_passed = [x + 1 for x in passed]
    print(res_passed)
    return dis


if __name__ == "__main__":
    inf = 10086
    mgraph0 = [
        [0, 1, 12, inf, inf, inf],
        [inf, 0, 9, 3, inf, inf],
        [inf, inf, 0, inf, 5, inf],
        [inf, inf, 4, 0, 13, 15],
        [inf, inf, inf, inf, 0, 4],
        [inf, inf, inf, inf, inf, 0],
    ]
    dis0 = startwith(0, mgraph0)
    print(dis0)
