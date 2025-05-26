from colorama import Fore
import json
import numpy as np
import os
from osgeo import gdal
import math
import cv2
from matplotlib import pyplot as plt
import time
import statistics
import threading
from Geo2Img import Geo2Img
from summon import summon
from PointsFilterRANSAC import PointsFilterRANSAC
from GetHeight import GetHeight
from PointAttainer import PointAttainer
from Img2Geo import Img2Geo
from GetAngle import GetAngle
from pyproj import transform, Proj
import warnings

warnings.filterwarnings("ignore")       # 忽视警告 

gdal.DontUseExceptions()                # 不管来自gdal的错误信息

################写入路径###############
RefImgPath = r'D:\WorkPlace\dingwei\shujuku2\jizhuntu\zhonghu_17level.tif'
DroneImgPath = r'D:\WorkPlace\dingwei\shujuku2'
######################################

################相机内参##############
K = np.eye(3)
#####################################

################写入初始经纬度#########
lon = 108.9669867
lat = 34.15534111
AssumingRatio = 0.1
######################################

#################################################################################################
################录入遥感图#############
RefImg = cv2.imread(RefImgPath)
if RefImg.shape[2] == 4:
    RefImg = RefImg[:, :, :3]
######################################

################生成或读取遥感特征信息##############
KPath = r"D:\WorkPlace\Program(1)\UnitV2\keypoints.json"
DPath = r"D:\WorkPlace\Program(1)\UnitV2\descriptors.npy"
if os.path.exists(KPath) and os.path.exists(DPath):
    with open(r'D:\WorkPlace\Program(1)\UnitV2\keypoints.json', 'r') as f:
        keypoints_data = json.load(f)

    Rd = np.load(r'D:\WorkPlace\Program(1)\UnitV2\descriptors.npy')
    Rp = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2], 
                          response=kp[3], octave=kp[4], class_id=kp[5]) for kp in keypoints_data]
else:
    Rp, Rd = summon(RefImg)
    keypoints_data = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in Rp]
    with open('keypoints.json', 'w') as f:
        json.dump(keypoints_data, f)

    np.save('descriptors.npy', Rd)

######################################

################得到遥感图地理信息#############
RefImgGeo = gdal.Open(RefImgPath)
TiffTransform = RefImgGeo.GetGeoTransform()
TIffCRS = RefImgGeo.GetProjection()
##############################################

################得到初始Tiff初始点#############
Coo = Geo2Img(TIffCRS, TiffTransform, lon, lat)
##############################################

########统计无人机拍摄图像数量，生成列表#########
framenum = 0
filename = os.listdir(DroneImgPath)
JPGName = []
for file in filename:
    if file.endswith('.JPG'):
        JPGName.append(file)
        framenum += 1                                                   
framenum = len(JPGName)

result = np.zeros((framenum, 2))

degree = np.zeros((framenum, 3))

height = np.zeros((framenum, 1))

lonlat = np.zeros((framenum, 2))

Time = np.zeros((framenum, 1))
'''
result矩阵各列说明: 0、图片序号 1、匹配关键字 2、偏航角 3、缩放比例 4、纬度 5、经度 
                   6、高度 7、匹配时间 8、图像中心x坐标 9、图像中心y坐标 10、图像物理对角线距离 11、图像像素对角线距离
'''
#################################################

####################随库匹配######################
i = 0
num = 0
trail = 0
size = 1000
CooToCrop = [Coo[0] - 500, Coo[1] - 750]

while i < framenum:
    start = time.perf_counter()
       
    result[i, 0] = i + 1
    if i == 0:
        RpHere, RdHere = PointAttainer(CooToCrop, 1000, 1500, (Rp, Rd))
    else:
        RpHere, RdHere = PointAttainer(CooToCrop, size, size, (Rp, Rd))
        
    DImg = cv2.imread(os.path.join(DroneImgPath, JPGName[i]))
    row0, col0, _ = DImg.shape
    row = int(row0 * AssumingRatio)
    col = int(row0 * AssumingRatio)
    DImg = cv2.resize(DImg, (col, row), interpolation=cv2.INTER_AREA)
    DImg = cv2.GaussianBlur(DImg, (5, 5), 0)
    Mp, Md = summon(DImg)                                                  # 开辟另一个线程处理
 
###############################################################################
    CenterCooInDroneImg = np.array([[row/2], [col/2], [1]])
    try:
        ImgTransfrom, matchesMask, UsablePT = PointsFilterRANSAC((RpHere, RdHere), (Mp, Md))
        ImgTransfrom /= np.linalg.norm(ImgTransfrom)
        
        end = time.perf_counter()
        print(f"消耗时间:{(end-start):.3f}s")
        # 检查匹配点的数量
        
        SemiUpdatedCooTotal = np.dot(ImgTransfrom, CenterCooInDroneImg)
        Coo = [int(SemiUpdatedCooTotal[0]/SemiUpdatedCooTotal[2]), int(SemiUpdatedCooTotal[1]/SemiUpdatedCooTotal[2])]
        CooToCrop = [Coo[0] - 250, Coo[1] - 250]
        result[i, 1] = 1
        num += 1
            
        degree[i] = GetAngle(ImgTransfrom, K)         
        f = 3986.018
        height[i] = GetHeight(f, ImgTransfrom, CenterCooInDroneImg)
        if i > 0 and abs(height[i] - height[i-1]) > 50:
            height[i] = height[i-1] 
        lon, lat = Img2Geo(TIffCRS, TiffTransform, Coo)
        lonlat[i, 0] = lon
        lonlat[i, 1] = lat
    
        #draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)
        #output = cv2.drawMatches(RefImg, RpHere, DImg, Mp, UsablePT, None, **draw_params)
        #plt.imshow(output), plt.show()
        
        print(Fore.GREEN + f"第{i+1}张图片匹配成功")
        i += 1
        size = 1000
        trail = 0
    
    except Exception:
        if trail == 0:
            size = 00
            trail = 1
        elif trail == 1:
            size = 800
            trail = 2
        else:
            result[i, 1] = 0
            print(Fore.RED + f"第{i+1}张图片匹配失败")
            trail = 0
            size = 800
            i += 1

######################################评估部分#############################################
LonPath = r'D:\WorkPlace\dingwei\test\test2\reference_longitude.txt'
LatPath = r'D:\WorkPlace\dingwei\test\test2\reference_latitude.txt'
heightPath = r'D:\WorkPlace\dingwei\test\test2\reference_height.txt'
HeadingPath = r'D:\WorkPlace\dingwei\test\test2\reference_heading_camera.txt'

Londata = np.loadtxt(LonPath)
Latdata = np.loadtxt(LatPath)
heightdata = np.loadtxt(heightPath)
Headingdata = np.loadtxt(HeadingPath)

print()

# 提前灰度化
# 一次性提取特征点
# 特征对质量评估
# 作图
# heading问题
# 显示画箭头，图位置

# 了解各个图像匹配算法
# 打印所有结果

import pandas as pd

# 创建DataFrame来存储经纬度和高度
data = {
    '经度': lonlat[:, 0],
    '纬度': lonlat[:, 1],
    '高度': height[:, 0]
}

df = pd.DataFrame(data)
# 确定桌面路径
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

# 保存DataFrame到CSV文件
csv_path = os.path.join(desktop_path, 'results.csv')
df.to_csv(csv_path, index=False)

print(f"结果已保存到 {csv_path}")