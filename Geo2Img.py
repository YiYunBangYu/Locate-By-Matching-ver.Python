from pyproj import transform , Proj
def Geo2Img( crs , mat , lon , lat ):
    xProj , yProj = transform( Proj(init = 'epsg:4326') , Proj(crs) , lon , lat)
    x = abs( int((xProj - mat[0]) / mat[1]) )
    y = abs( int((yProj - mat[3]) / mat[5]) )
    Coo = [ x , y ] 
    return Coo