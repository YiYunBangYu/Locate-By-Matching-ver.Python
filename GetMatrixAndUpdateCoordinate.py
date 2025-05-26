from cv2 import estimateAffine2D
from numpy import dot

def GetMatrixAndUpdateCoordinate(RefPT , MatPT , OriginCoo , CenterCoordinateInDroneImg):
    transform , _ = estimateAffine2D(RefPT , MatPT)
    UpdatedPointNP = dot(transform , CenterCoordinateInDroneImg)
    UpdatedCoo = [int(UpdatedPointNP[0] + OriginCoo[0]) , int(UpdatedPointNP[1] + OriginCoo[1])]
    return transform , UpdatedCoo
    
    
    