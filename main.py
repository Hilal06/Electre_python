import math
import numpy as np

def normalized_decision(data: [], root_arr: []):
    criteria = data
    root = []
    normal = []
    for i, row in enumerate(root_arr):
        tmp = 0
        for col in row:
            tmp += math.pow(col, 2)
        root.append(math.sqrt(tmp))

    for row in criteria:
        tmp = []
        for index, col in enumerate(row):
            res = col/root[index]
            tmp.append(res)
        normal.append(tmp)

    return np.array(normal)


def weighted_normalized(matrix: [], weight: []):
    result = []
    for row in matrix:
        tmp = []
        for i, col in enumerate(row):
            res = col * weight[i]
            tmp.append(res)
        result.append(tmp)
    print(np.array(result))
    return result


def indexing_concordance(data: [], x: int, y: int):
    matrix = []
    tmp_a = data[x-1]
    tmp_b = data[y-1]
    for i, col in enumerate(tmp_a):
        res = 1 if col >= tmp_b[i] else 0
        matrix.append(res)
    return matrix

def indexing_discordance(data: [], x: int, y: int):
    matrix = []
    tmp_a = data[x - 1]
    tmp_b = data[y - 1]
    for i, col in enumerate(tmp_a):
        res = 1 if col < tmp_b[i] else 0
        matrix.append(res)
    return matrix

def index_table(data: [], type: str):
    result = []
    for i, row1 in enumerate(data):
        for j, row2 in enumerate(data):
            x = i+1
            y = j+1
            if x is not y:
                if type is "concordance":
                    tmp: [] = indexing_concordance(data, x, y)
                    idt = (x, y)
                    result.append([idt, tmp])
                else:
                    tmp: [] = indexing_discordance(data, x, y)
                    idt = (x, y)
                    result.append([idt, tmp])
    return np.array(result)

def concordance(data: [], w: []):
    result = []
    for x, row1 in enumerate(data):
        tmp = []
        for y, row2 in enumerate(data):
            if x is y:
                tmp.append(0)
            else:
                tmp_arr = indexing_concordance(data, x+1, y+1)
                sum = 0
                for i, cell in enumerate(tmp_arr):
                    sum += 0 if cell is 0 else w[i]
                tmp.append(sum)
        result.append(tmp)
    print(result)
    return result

def discordance(data: []):
    result = []
    for x, row in enumerate(data):
        for y, row2 in enumerate(data):
            if x != y:
                tmp = []
                tmp2 = []
                tmp_res = []
                for k in range(len(data[0])):
                    idx = indexing_discordance(data, x+1, y+1)
                    diff = abs(data[x][k] - data[y][k])
                    if idx[k] == 1:
                        disc = diff
                    else:
                        disc = 0
                    tmp.append(disc)
                    tmp2.append(diff)
                max1 = max(tmp)
                max2 = max(tmp2)
                res = max1/max2
                tmp_res.append(res)
                result.append(tmp)
    return result


def threshold(matrix: []):
    total = 0
    count = 0
    for row in matrix:
        for col in row:
            total += col
        count += 1
    return total / (count*(count-1))

def dominan_concordance(cor: [], threshold: float):
    res = []
    for row in cor:
        tmp = []
        for col in row:
            x = 1 if col > threshold else 0
            tmp.append(x)
        res.append(tmp)
    print("Concordance: ", np.array(res))

def main():
    data_arr = [
        [2, 4, 2, 2, 3],
        [4, 1, 5, 5, 3],
        [3, 2, 1, 4, 4]
    ]
    weight = [3, 2, 2, 2, 1]
    data = np.array(data_arr)
    norm = normalized_decision(data, data.transpose())
    print("Normalized : ")
    print(norm)
    norm_weight = weighted_normalized(norm, weight)
    print("Weighted Normalized : ", norm_weight)

    index_tables = index_table(norm_weight, "concordance")
    print("index concordance:", index_tables)
    index_tables = index_table(norm_weight, "discordance")
    print("index discordance: ", index_tables)

    cor = concordance(data_arr, weight)
    print("concordance:", cor)
    print("===================================")
    dis = discordance(norm_weight)
    print(np.array(dis))
    print("===================================")

    trs = threshold(cor)
    print("Threshold:", trs)
    dominan_concordance(cor, trs)


main()

