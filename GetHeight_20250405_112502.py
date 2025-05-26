from numpy import sqrt, dot, array

def GetHeight(f, transform, Coo):
    # 使用变换矩阵将坐标转换为中心坐标
    Center_Coo = dot(transform, Coo)
    Center_Coo = [Center_Coo[0] / Center_Coo[2], Center_Coo[1] / Center_Coo[2]]
    
    # 创建一个稍微偏移的坐标点
    temp = Coo + array([[0.1], [0.1], [0]])
    Temp_Coo = dot(transform, temp)  # 使用新的变量名 Temp_Coo
    Temp_Coo = [Temp_Coo[0] / Temp_Coo[2], Temp_Coo[1] / Temp_Coo[2]]
    
    # 计算两个中心坐标点之间的距离
    dx = Temp_Coo[0] - Center_Coo[0]  # 修正 dx 的计算
    dy = Temp_Coo[1] - Center_Coo[1]  # 修正 dy 的计算
    h = sqrt(dx**2 + dy**2)
    
    # 应用焦距进行高度计算
    h *= f
    
    return h

