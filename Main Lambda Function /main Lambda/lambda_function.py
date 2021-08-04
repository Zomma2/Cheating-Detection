import json
import os
import boto3
s3 = boto3.resource('s3')
import numpy as np
from pose_estimation import pose_fns
from pose_estimation import predictor
from objDet import yolo 
import sys
sys.path.append('/mnt/acess')
import pandas as pd
lam = boto3.client('lambda')
import cv2
import time
import requests 

def lambda_handler(event, context):
    
    report_payload={}
    '''
    #step 1 : 
    #Download object detection files
    bucket = 'object-detection-grad-proj'
    key_1= 'data/yolo-coco/yolov3.weights'
    key_2 = 'data/yolo-coco/yolov3.cfg'
    key_3 = 'data/yolo-coco/coco.names'
    s3.meta.client.download_file(
        bucket,key_2,'/mnt/acess/yolov3.cfg')
    s3.meta.client.download_file(
        bucket,key_3,'/mnt/acess/coco.names')
    s3.meta.client.download_file(
        bucket,key_1,'/mnt/acess/yolov3.weights')
        
    '''
    '''
    #step 2 : 
    #Download test image
    bucket = 'object-detection-grad-proj'
    key= 'data/images/soccer.jpg'
    s3.meta.client.download_file(
        bucket,key,'/mnt/acess/soccer.jpg')    

    '''
    '''
    #step 3 : 
    #Download palgarism_objects files
    bucket = 'object-detection-grad-proj'
    key= 'data/palgarism_objects_to_detect.txt'
    s3.meta.client.download_file(
        bucket,key,'/mnt/acess/palgarism_objects_to_detect.txt')  

    bucket = 'object-detection-grad-proj'
    key= 'data/palgarism_objects.txt'
    s3.meta.client.download_file(
        bucket,key,'/mnt/acess/palgarism_objects.txt')  
    '''
    
    # step 4 
    # try yolo
    '''
    output =yolo.detect("/mnt/acess/soccer.jpg","/mnt/acess/")
    filtered_out = yolo.filter_output_yolo(output ,  yolo.get_palgarism_objects_txt("/mnt/acess/palgarism_objects.txt",2))
    '''
    
    
    
    
    '''
    Before Audio Model takes place 
    
    bucket='lamda-grad-proj-bucket'
    imgs_dir_list = s3_efs.sync_s3_to_efs(bucket) ## sync efs with s3 and get directories of imgs
    for img_dir in imgs_dir_list : 
        images = os.listdir(img_dir)
        for image in images : 
            cv2.imread(image)
    '''
    
    '''
    X = ['center','up','center','down','up']
    J = ['left','right','center','center','right']
    xx= pose_fns.is_cheating_pose(X,J,1)
    '''
    
    '''
    bucket = 'object-detection-grad-proj'
    key= 'stud_1.wav'
    s3.meta.client.download_file(
        bucket,key,'/mnt/acess/stud_1.wav')   
    '''
    

    '''
    image = cv2.imread('/mnt/acess/student_1/Photo on 05-01-2021 at 6.42 AM #6.jpg')
    result = predictor.predict(image ,ep )
    '''
    
    
    
    
    ep = 'pytorch-inference-eia-2021-07-23-15-25-51-547'
    
     
    
    def get_array_from_txt(txt_file,split = ','):
        f = open(txt_file, "r")
        arr = f.read().split(split)
        f.close()
        return arr[:-1]
        
    def parse_voice_values (txt_file) : 
        a,b,c = get_array_from_txt(txt_file,',,')
        a = [eval(i) for i in a[1:-1].split(',')]
        b = np.int64(b)
        c= eval(c)
        return a,b,c 



    def atoi(text):
        return int(text) if text.isdigit() else text
        
    
    def natural_keys(text):
         

        
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    
    
    def is_cheating_pose_yolo_detection_run(path , threshold) : 
        pitch_pose_list=[]
        yaw_pose_list=[]
        all_yolo_obj_list=[]
        all_yolo_obj_result=[]
        student_image_list = []
        failure_frames_list=[]
        pitch = None  
        yaw = None
        is_exist = None
        for i in range (1,8) : 
            frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
            print(path+'/'+'{}.jpeg'.format(i))
            is_exist = os.path.isfile(path+'/'+'{}.jpeg'.format(i))
            print('File exist ?',is_exist)
            print(len(os.listdir(path + '/')))
            try :
                print(frame.shape)
            except : 
                print('cannot read image 1')
                print('File exist ?',os.path.isfile(path+'/'+'{}.jpeg'.format(i)))
                frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                try :
                    print(frame.shape)
                except : 
                    if not is_exist :
                        time.sleep(0.8) 
                    print('cannot read image 2' )
                    frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                    try : 
                        
                        print(frame.shape)
                    except :
                        frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                        try :
                            print(frame.shape)
                        except : 
                            key_search = path.split('/')[-1] +'/'+'{}.jpeg'.format(i)
                            if not is_exist :
                                time.sleep(0.5)
                            frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                            try : 
                                print(frame.shape)
                            except :
                                frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                                try :
                                    print(frame.shape)
                                except:
                                    failure_frames_list.append(i)
                                    continue
            try : 
                pitch , yaw, _ = predictor.predict(frame,ep)
                yaw , pitch = pose_fns.transfer_to_directions(yaw , pitch)
            except : 
                yaw = 'right'
                pitch = 'up'
            try :
                yolo_obj_list ,yolo_obj_result  = yolo.is_cheating_yolo(path+'/'+'{}.jpeg'.format(i))
                pitch_pose_list.insert(i,pitch)
                yaw_pose_list.append(yaw)
                
                all_yolo_obj_list.append(yolo_obj_list)
                all_yolo_obj_result .append(yolo_obj_result)
            except : 
                continue
            ts = time.time()
            try : 
                os.rename(path+'/'+'{}.jpeg'.format(i) ,path+'/'+ '{}.jpeg'.format(str(ts)))
                student_image_list.append(path+'/'+ '{}.jpeg'.format(str(ts)))
            except : 
                print('Naming process is intrupted by another writing process')
                print('Retrying Again to rename')
                try : 
                    os.rename(path+'/'+'{}.jpeg'.format(i) ,path+'/'+ '{}.jpeg'.format(str(ts)))
                    student_image_list.append(path+'/'+ '{}.jpeg'.format(str(ts)))
                except : 
                    print("Failed to backup renaming process")
                    continue
    
        if  len(failure_frames_list) > 0 :     
            print("Trying To fetch unproceessed frames")
            for i in failure_frames_list : 
                frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                print(path+'/'+'{}.jpeg'.format(i))
                print('File exist ?',os.path.isfile(path+'/'+'{}.jpeg'.format(i)))
                print(len(os.listdir(path + '/')))
                try :
                    print(frame.shape)
                except : 
                    print("Trying To fetch unproceessed frames")
                    print('cannot read image 1')
                    if not is_exist :
                        time.sleep(1)
                    print('File exist ?',os.path.isfile(path+'/'+'{}.jpeg'.format(i)))
                    frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                    try :
                        print(frame.shape)
                    except : 
                        print("Trying To fetch unproceessed frames")
                        print('cannot read image 2' )
                        key_search = path.split('/')[-1] +'/'+'{}.jpeg'.format(i)
                        try : 
                            print(frame.shape)
                        except :
                            if not is_exist :
                                time.sleep(1)
                            frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                            try :
                                print(frame.shape)
                            except : 
                                key_search = path.split('/')[-1] +'/'+'{}.jpeg'.format(i)
                                try : 
                                    s3.meta.client.download_file('lamda-grad-proj-bucket',key_search+'/'+'{}.jpeg'.format(i),path+'/'+'{}.jpeg'.format(i))
                                except :
                                    frame = cv2.imread(path+'/'+'{}.jpeg'.format(i))
                                    try :
                                        print(frame.shape)
                                    except:
                                        continue
                try : 
                    pitch , yaw, _ = predictor.predict(frame,ep)
                    yaw , pitch = pose_fns.transfer_to_directions(yaw , pitch)
                except : 
                    yaw = 'right'
                    pitch = 'up'
                try : 
                    yolo_obj_list ,yolo_obj_result  = yolo.is_cheating_yolo(path+'/'+'{}.jpeg'.format(i))
                    pitch_pose_list.insert(i-1 , pitch)
                    yaw_pose_list.insert(i-1 ,yaw)
        
                    all_yolo_obj_list.insert(i-1 ,yolo_obj_list)
                    all_yolo_obj_result.insert(i-1 ,yolo_obj_result)
                except :
                    continue
                ts = time.time()
                try : 
                    os.rename(path+'/'+'{}.jpeg'.format(i) ,path+'/'+ '{}.jpeg'.format(str(ts)))
                    student_image_list.insert(i-1 ,path+'/'+ '{}.jpeg'.format(str(ts)))
                except : 
                    print('Naming process is intrupted by another writing process')
                    print('Retrying Again to rename')
                    try : 
                        os.rename(path+'/'+'{}.jpeg'.format(i) ,path+'/'+ '{}.jpeg'.format(str(ts)))
                        student_image_list.insert(i-1 ,path+'/'+ '{}.jpeg'.format(str(ts)))
                    except : 
                        print("Failed to backup renaming process")
                        continue
    
                
        print(pitch_pose_list ,yaw_pose_list ,all_yolo_obj_result,all_yolo_obj_list,student_image_list)
        pose_calc_V,pose_calc_H,cheating_score ,cheating_rate ,  cheating_result = pose_fns.is_cheating_pose(pitch_pose_list,yaw_pose_list,threshold)
        report_payload['student_image_list'] = student_image_list
        report_payload['pose_calc_H'] = pose_calc_H 
        report_payload['pose_calc_V'] = pose_calc_V 
        report_payload ['path'] = path
        return cheating_score ,cheating_rate ,  cheating_result ,all_yolo_obj_list,all_yolo_obj_result
        
    
    def cheating_detection_complete_run(path , score_threshold , sound_threshold) : 
        detected_frames_list = []
        sound_result_list = [] 
        cheating_score ,cheating_rate ,  cheating_result ,all_yolo_obj_list,all_yolo_obj_result = is_cheating_pose_yolo_detection_run(path , score_threshold)
        time.sleep(1)
        file = path+'/vad.txt'
        try :
            is_speech_np, detected_frames , sound_result = parse_voice_values(file)
        except : 
            time.sleep(1)
            is_speech_np, detected_frames , sound_result = parse_voice_values(file)
            
        report_payload['wav_file_path'] = path+'/' 
        report_payload ['is_speech_np'] = is_speech_np           
        detected_frames_list.append(detected_frames)
        sound_result_list.append(sound_result)
        return cheating_score ,cheating_rate ,cheating_result ,all_yolo_obj_list,np.sum(all_yolo_obj_result) ,detected_frames_list,sound_result_list 
            
        
    
    def cheating_detection_reporting_and_responsing(path,score_threshold,frames_threshold): 
        cheating_score ,cheating_rate ,cheating_result ,all_yolo_obj_list,all_yolo_obj_result ,detected_frames_list\
        ,sound_result_list = cheating_detection_complete_run(path,score_threshold,frames_threshold)
        report_df = pd.DataFrame ({
        'cheating score':cheating_score ,
        'cheating rate':cheating_rate ,
        'Pose cheating result' : cheating_result ,
        'objects_detected' :str(all_yolo_obj_list),
        'Num objects detected' : all_yolo_obj_result,
        'Num Frames of Human sound':detected_frames_list,
        'Sound Detection result':sound_result_list },index = ['Student']).T
        
        if cheating_result or all_yolo_obj_result or np.array(sound_result_list).sum() : 
            return all_yolo_obj_list, True , report_df,cheating_result , all_yolo_obj_result , np.array(sound_result_list).sum()
        else :
            return all_yolo_obj_list , False , report_df , False , False ,False
    
    
    path = '/mnt/acess/'+event['path']
    all_yolo_obj_list ,result , report_df,result_pose , result_yolo , result_sound  = cheating_detection_reporting_and_responsing(path , 0.55,0.6)
    
    


    f = open( path+"/type.txt", "r")
    Type = f.read()
    print(Type)
    f.close()
    print(len(all_yolo_obj_list))
    print(result_pose or  result_yolo or  result_sound)
    
    if len(all_yolo_obj_list) > 3 : 
        if result == True : 
            report_payload['report_df'] = report_df.values.tolist()
            report_payload['stud_name'] =  event['path'].split(':')[0]
            Resp = lam.invoke(FunctionName='Reporting_Lambda',
            InvocationType='Event',
            Payload=json.dumps(report_payload))
             
             
         
    if Type == "online" : 
        if result_pose or  result_yolo or  result_sound :
            print('online')
            x = requests.post('https://bad-frog-71.loca.lt/cheatingDetection/getResult', json = {'userId':'6071299a5bf57641c44f84bd',
            'examId' : '1' , 'username' : '2'
            })
            print(x)
             
          
    if Type == "file"  :
        if  result_yolo or  result_sound :
            print('file')
            x = requests.post('https://bad-frog-71.loca.lt/cheatingDetection/getResult', json = {'userId':'6071299a5bf57641c44f84bd',
            'examId' : '1' , 'username' : '2'
            })
            print(x)
            
     
                 
    




    
    return {
        'statusCode': 200
    }
    










'''
https://stackoverflow.com/questions/35455281/aws-lambda-how-to-set-up-a-nat-gateway-for-a-lambda-funstud_namection-with-vpc-access
'''
