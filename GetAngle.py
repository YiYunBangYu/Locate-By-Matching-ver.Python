import numpy as np

def GetAngle(Matrix , K):
    Matrix_normalized = np.dot(np.linalg.inv(K) , Matrix)
    
    R_1 = Matrix_normalized[:,0]
    R_2 = Matrix_normalized[:,1]
    
    R_1 /= np.linalg.norm(R_1)
    R_2 /= np.linalg.norm(R_2)
    R_3 = np.cross(R_1,R_2)
    R = np.column_stack((R_1, R_2, R_3))
    
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[2, 1], R[2, 2])
    roll = np.arctan2(R[1, 0], R[0, 0])
        
    degree = np.zeros(shape=(1,3))

    degree[0,0] = pitch
    degree[0,1]= yaw
    degree[0,2] = roll
    
    i = 0
    while i < 2:
        degree[0,i] = np.degrees(degree[0,i])
        
        if degree[0,i] > 180:
            degree[0,i] -= 360
        if degree[0,i] < -180:
            degree[0,i] += 360

        i+=1
    
    return degree