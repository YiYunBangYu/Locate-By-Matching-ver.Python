from pyproj import transform , Proj

def Img2Geo( crs , mat , Coo ):
    xProj = mat[0] + Coo[0] * mat[1] + Coo[1] * mat[2]
    yProj = mat[3] + Coo[0] * mat[4] + Coo[1] * mat[5]
    lon , lat = transform( Proj(crs) , Proj(init = 'epsg:4326') , xProj , yProj )
    return lon , lat