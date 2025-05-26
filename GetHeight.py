from numpy import sqrt, dot, array, linalg
from scipy.optimize import least_squares

def Geo2Img(crs, transform, lon, lat):
    # 假设的函数，将地理坐标转换为图像坐标
    return [lon, lat, 1]

def reprojection_error(params, transform, gcp_img, gcp_world):
    inv_transform = array(params).reshape(3, 3)
    gcp_world_pred = dot(inv_transform, gcp_img) / inv_transform[2, 2]
    return gcp_world_pred[:2] - gcp_world

def GetHeight(f, transform, Coo, GCPs=None):
    """
    计算高度
    
    参数:
    f: 相机焦距
    transform: 单应性矩阵
    Coo: 图像中心点坐标 (3x1 矩阵)
    GCPs: 可选的地面控制点列表，包含 (经度, 纬度, 高度) 三元组
    """
    # 计算图像中心点在世界坐标系中的坐标
    CenterCoo = dot(transform, Coo)
    CenterCoo = [CenterCoo[0] / CenterCoo[2], CenterCoo[1] / CenterCoo[2]]
    
    # 假设在图像坐标中移动 [0.01, 0.01] 像素
    temp = Coo + array([[0.01], [0.01], [0]])
    CenterCoo_ = dot(transform, temp)
    CenterCoo_ = [CenterCoo_[0] / CenterCoo_[2], CenterCoo_[1] / CenterCoo_[2]]
    
    dx = CenterCoo_[0] - CenterCoo[0]
    dy = CenterCoo_[1] - CenterCoo[1]
    
    # 计算图像中的像素距离
    pixel_distance = sqrt(dx**2 + dy**2)
    
    ground_distance = 0
    
    # 如果有地面控制点，可以使用它们来计算更准确的高度
    if GCPs is not None and len(GCPs) >= 2:
        # 收集图像坐标和世界坐标
        image_points = []
        world_points = []
        
        for gcp in GCPs:
            img_coo = Geo2Img(TIffCRS, TiffTransform, gcp[0], gcp[1])
            image_points.append(img_coo)
            world_points.append([gcp[0], gcp[1]])
        
        image_points = array(image_points).reshape(-1, 3)
        world_points = array(world_points).reshape(-1, 2)
        
        # 使用最小二乘法优化单应性矩阵的逆
        initial_guess = transform.flatten()
        result = least_squares(reprojection_error, initial_guess, args=(transform, image_points, world_points))
        optimized_inv_transform = result.x.reshape(3, 3)
        
        # 计算多个地面控制点之间的实际地面距离和像素距离
        total_actual_distance = 0
        total_pixel_distance = 0
        
        for i in range(len(GCPs) - 1):
            gcp1 = GCPs[i]
            gcp2 = GCPs[i + 1]
            
            # 计算两个GCPs之间的实际地面距离
            actual_distance = sqrt((gcp2[0] - gcp1[0])**2 + (gcp2[1] - gcp1[1])**2)
            
            # 计算两个GCPs之间的图像像素距离
            gcp1_img = Geo2Img(TIffCRS, TiffTransform, gcp1[0], gcp1[1])
            gcp2_img = Geo2Img(TIffCRS, TiffTransform, gcp2[0], gcp2[1])
            
            gcp1_img = array(gcp1_img).reshape(3, 1)
            gcp2_img = array(gcp2_img).reshape(3, 1)
            
            gcp1_world = dot(optimized_inv_transform, gcp1_img)
            gcp2_world = dot(optimized_inv_transform, gcp2_img)
            
            gcp1_world = [gcp1_world[0] / gcp1_world[2], gcp1_world[1] / gcp1_world[2]]
            gcp2_world = [gcp2_world[0] / gcp2_world[2], gcp2_world[1] / gcp2_world[2]]
            
            pixel_distance_gcp = sqrt((gcp2_world[0] - gcp1_world[0])**2 + (gcp2_world[1] - gcp1_world[1])**2)
            
            # 累加实际距离和像素距离
            total_actual_distance += actual_distance
            total_pixel_distance += pixel_distance_gcp
        
        # 计算像素与实际距离的比例
        if total_pixel_distance != 0:
            pixel_to_ground_ratio = total_actual_distance / total_pixel_distance
        else:
            pixel_to_ground_ratio = 1  # 避免除以零
        
        # 使用比例计算高度
        ground_distance = pixel_distance * pixel_to_ground_ratio
    
    # 计算高度
    h = ground_distance * f
    
    return h
