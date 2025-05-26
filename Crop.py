def Crop(img , Coordinate , V , H):
    h = img.shape[0]
    w = img.shape[1]
    
    Cx = Coordinate[0]
    Cy = Coordinate[1]
    
    if Cy + V > h:
        V = h - Cy
    if Cx + H > w:
        H = w - Cx
        
    img = img[Cy: Cy + V , Cx : Cx + H]
    
    return img , V , H