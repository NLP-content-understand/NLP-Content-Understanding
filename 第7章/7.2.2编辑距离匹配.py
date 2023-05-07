def Distance_Recursive(str1, str2):
    if len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    elif str1 == str2:
        return 0

    if str1[len(str1) - 1] == str2[len(str2) - 1]:
        d = 0
    else:
        d = 1

    return min(Distance_Recursive(str1, str2[:-1]) + 1,
               Distance_Recursive(str1[:-1], str2) + 1,
               Distance_Recursive(str1[:-1], str2[:-1]) + d)


if __name__ == '__main__':
    minDistance = float('inf')#无穷大
    # Dict为标准词语库 

    Dict = ['字典库']
    input = "你的输入"
    for i in Dict:
        if (Distance_Recursive(i, input)) < minDistance:
            minDistance = Distance_Recursive(i, input)
            print(i)
