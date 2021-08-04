import numpy as np

def calc_pose_score(list_of_pose):
    score = []
    for i , x in enumerate(list_of_pose) : 
        if i == 0 :
            if x == 'center' : 
                score.append(0)
            else : 
                score.append(1)
        if i > 0 : 
            j = i - 1 
            if x == 'center' : 
                score.append(0)
            else : 
                if list_of_pose[j] == 'center':
                    score.append(1)
                if list_of_pose[j] == x:
                    score.append(1)
                if list_of_pose[j] != x and list_of_pose[j] != 'center' :
                    score.append(2)
    return score 
    
    
def get_pose_final_score (V_list_of_pose , H_list_of_pose) : 
    pose_calc_V = calc_pose_score(V_list_of_pose)
    pose_calc_H = calc_pose_score(H_list_of_pose)
    return pose_calc_V,pose_calc_H,np.sum(np.array(pose_calc_V)+np.array(pose_calc_H))
                          
                          
def is_cheating_pose (V_list_of_pose , H_list_of_pose,threshold) : 
    pose_calc_V,pose_calc_H,cheating_score= get_pose_final_score(V_list_of_pose , H_list_of_pose)
    cheating_rate = cheating_score / len(H_list_of_pose)
    cheating_result = cheating_rate > threshold 
    return pose_calc_V,pose_calc_H, cheating_score ,cheating_rate ,  cheating_result

def transfer_to_directions(yaw , pitch ) :
    dir_yaw = '' 
    dir_pitch = '' 
    if -12 < yaw < 12 : 
        dir_yaw = 'center'
    elif yaw < -12 : 
        dir_yaw = 'right'
    elif yaw > 12 : 
        dir_yaw = 'left'
    if -16 < pitch < 16 : 
        dir_pitch = 'center'
    elif pitch < -16: 
        dir_pitch = 'down'
    elif pitch > 16 : 
        dir_pitch = 'up'
    return dir_yaw , dir_pitch 


 