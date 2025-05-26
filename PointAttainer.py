import numpy as np

def PointAttainer( Coo , col , row , ImgInformation ):
    
    lx = Coo[0]
    rx = Coo[0] + col
    uy = Coo[1]
    dy = Coo[1] + row
    
    pHere = []    
    dHere = []

    for i , point in enumerate(ImgInformation[0]): 
        if point.pt[0]> lx and point.pt[0] < rx and point.pt[1] > uy and point.pt[1] < dy:
            pHere.append(point)
            dHere.append(ImgInformation[1][i])
    
    dHere = np.array(dHere)
    dHere = dHere.astype(np.float32)
    
    return  pHere , dHere
    
            
    