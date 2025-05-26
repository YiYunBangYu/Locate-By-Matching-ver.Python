from cv2 import FlannBasedMatcher , findHomography , RANSAC
from numpy import float32

def PointsFilterRANSAC(RInformation , MInformation):
    Index = dict(algorithm = 1 , trees = 5 )
    Search = dict(check = 50)                                               

    flann = FlannBasedMatcher(Index, Search)
    matches = flann.knnMatch(RInformation[1] , MInformation[1] , k = 2)
    UsablePT = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            UsablePT.append(m)                                              
    RefPT = float32([RInformation[0][m.queryIdx].pt for m in UsablePT]).reshape(-1, 1, 2)
    MatPT = float32([MInformation[0][m.trainIdx].pt for m in UsablePT]).reshape(-1, 1, 2) 
    
    M, mask = findHomography(MatPT, RefPT, RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return M , matchesMask , UsablePT