from colorama import Fore
import numpy as np
import os
from osgeo import gdal
import math
import cv2
from matplotlib import pyplot as plt
import time
import statistics
from Geo2Img import Geo2Img
from Crop import Crop
from summon import summon
from PointsFilterRANSAC import PointsFilterRANSAC
from GetHeight import GetHeight
from Img2Geo import Img2Geo
from pyproj import transform , Proj
import warnings

warnings.filterwarnings("ignore")       #忽视警告 

gdal.DontUseExceptions()                #不管来自gdal的错误信息
################写入路径###############
RefImgPath = 'D:\\WorkPlace\\About project\\Database2\\RefImg\\zhonghu_17level.tif'
DroneImgPath = 'D:\\WorkPlace\\About project\\Database2'
######################################

################写入初始经纬度#########
lon=108.9669867
lat=34.15534111
######################################
#################################################################################################
################录入遥感图#############
RefImg = cv2.imread(RefImgPath)
if RefImg.shape[2] == 4:
    RefImg = RefImg[:, :, :3]
######################################

################得到遥感图地理信息#############
RefImgGeo = gdal.Open(RefImgPath)
TiffTransform = RefImgGeo.GetGeoTransform()
TIffCRS = RefImgGeo.GetProjection()
##############################################

################得到初始Tiff初始点#############
Coo = Geo2Img(TIffCRS , TiffTransform , lon , lat )
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

result = np.zeros((framenum , 10))
'''
result矩阵各列说明: 0、图片序号 1、匹配关键字 2、偏航角 3、缩放比例 4、纬度 5、经度 
                   6、高度 7、匹配时间 8、图像中心x坐标 9、图像中心y坐标 10、图像物理对角线距离 11、图像像素对角线距离
'''
#################################################

####################随库匹配######################
i = 0
num = 0
trail = 0
size = 500
CooToCrop = [Coo[0] - 500 , Coo[1] - 750]

while i < framenum:
    start = time.perf_counter()
       
    result[i , 0] = i + 1
    if i == 0 :
        tempRef , _ , _ = Crop(RefImg , CooToCrop , 1500 , 1000)
    else:
        tempRef , _ , _ = Crop(RefImg , CooToCrop , size , size)
    RInformation = summon(tempRef)                              #Ref特征获取
    
    DImg = cv2.imread( os.path.join(DroneImgPath, JPGName[i]) )
    row0 , col0 , _ = DImg.shape
    AssumingRatio = 0.1
    row = int(row0 * AssumingRatio)
    col = int(row0 * AssumingRatio)
    DImg = cv2.resize(DImg , (col , row) , interpolation=cv2.INTER_AREA)
    MInformation = summon(DImg)
 
###############################################################################
    CenterCooInDroneImg = np.array([[row/2] , [col/2] , [1]])
    try:
        ImgTransfrom , matchesMask , UsablePT= PointsFilterRANSAC(RInformation , MInformation)
        ImgTransfrom /= np.linalg.norm(ImgTransfrom)
        
        end = time.perf_counter()
        print(f"消耗时间:{(end-start):.3f}s")
        
        SemiUpdatedCooTotal = np.dot(ImgTransfrom , CenterCooInDroneImg)
        SemiUpdatedCoo = [int(SemiUpdatedCooTotal[0]/SemiUpdatedCooTotal[2]) , int(SemiUpdatedCooTotal[1]/SemiUpdatedCooTotal[2])]
        Coo = [CooToCrop[0] + SemiUpdatedCoo[0] , CooToCrop[1] + SemiUpdatedCoo[1]]
        CooToCrop = [Coo[0] - 250 , Coo[1] - 250]
        result[i , 1] = 1
        num += 1
            
        angle = np.arctan2(ImgTransfrom[1 , 0] , ImgTransfrom[0 , 0])
        angleDegree = -np.degrees(angle)
        #if angleDegree > 180:
                #angleDegree -= 360
        #elif angleDegree < -180:
            #angleDegree += 360
        result[i , 2] = angleDegree
            
        f = 3986.018
        result[i,6] = GetHeight(f, ImgTransfrom , CenterCooInDroneImg)
        #if i > 0 and abs(result[i,6]-result[i-1,6]) > 0.1:
            #result[i,6] = result[i-1,6]
            
        lon , lat = Img2Geo(TIffCRS , TiffTransform , Coo)
        result[i,4] = lon
        result[i,5] = lat
            
        #draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)

        #output = cv2.drawMatches(tempRef, RInformation[0], DImg, MInformation[0], UsablePT, None, **draw_params)

        #plt.imshow(output), plt.show()
        print(Fore.GREEN + f"第{i+1}张图片匹配成功")
        i += 1
        size = 500
    
    except Exception:
        if trail == 0:
            size = 800
            trail = 1
        elif trail == 1:
            size = 1000
            trail = 2
        else:
            result[i,1] = 0
            print(Fore.RED + f"第{i+1}张图片匹配失败")
            trail = 0
            size = 500
            i += 1

######################################评估部分#############################################

##########################经纬度评估部分#########################
LonPath = 'D:\\WorkPlace\\About project\\reference_longitude.txt'
Londata = np.loadtxt(LonPath)

i = 0
LonEstimation = []
while i < framenum:
    if result[i , 1] != 0:
        LonEstimation.append(Londata[i] - result[i , 4])
    i += 1
print(f"\nLonEstimation方差:{statistics.variance(LonEstimation):.7f}\nLonEstimation平均值:{np.mean(LonEstimation):.3f}")
#################################################################
LatPath = 'D:\\WorkPlace\\About project\\reference_latitude.txt'
Latdata = np.loadtxt(LatPath)

i = 0
LatEstimation = []
while i < framenum:
    if result[i , 1] != 0:
        LatEstimation.append(Latdata[i] - result[i , 5])
    i += 1
print(f"LatEstimation方差:{statistics.variance(LatEstimation):.7f}\nLatEstimation平均值:{np.mean(LatEstimation):.3f}")
#################高度评估部分#######################################
heightPath = 'D:\\WorkPlace\\About project\\reference_height.txt'
heightdata = np.loadtxt(heightPath)

i = 0
R = []
while i < framenum:
    if result[i , 1] != 0:
        R.append(heightdata[i] - result[i , 6])
    i += 1
print(f"HeightEstimation方差:{statistics.variance(R):.7f}\nHeightEstimation平均值:{np.mean(R):.3f}\nErrorMaximum:{np.max(R):.3f}")
####################################################################

#################角度评估部分#####################################
HeadingPath = 'D:\\WorkPlace\\About project\\reference_heading_camera.txt'
Headingdata = np.loadtxt(HeadingPath)

i = 0
R = []
while i < framenum:
    if result[i , 1] != 0:
        R.append(Headingdata[i] / result[i , 2])
    i += 1
print(f"HeadingEstimation方差:{statistics.variance(R):.7f}\nHeadingEstimation平均值:{np.mean(R):.3f}")

##########################地理距离评估###############################


print()

#提前灰度化
#一次性提取特征点
#特征对质量评估
#作图
#heading问题
#显示画箭头，图位置

#了解各个图像匹配算法