import json
import os 
import shutil 
from RSM import RS
import sys 
sys.path.append('/mnt/acess/')
import pandas as pd
import boto3
import numpy as np 

lam = boto3.client('lambda')
def lambda_handler(event, context):

    
    student_image_list = event['student_image_list'] 
    pose_calc_H = event['pose_calc_H'] 
    pose_calc_V = event['pose_calc_V']
    path  = event['path']
    report_df =pd.DataFrame(event['report_df'],columns = ['Student'] , index = ['cheating score' ,
    'cheating rate' ,'Pose cheating result'  ,'objects_detected' ,'Num objects detected' ,'     Frames of Human sound (each 10 )','Sound Detection result'] )
    stud_name = event['stud_name'] 
    is_speech_np = np.array(event['is_speech_np'])
    wav_file_path = event['wav_file_path']
    RS.create_pdf_assets(student_image_list, pose_calc_H,pose_calc_V,path)
    RS.wafplot_VAD(is_speech_np,wav_file_path.replace(wav_file_path.split('/')[-1],''))
    RS.export_to_pdf (report_df , stud_name , path , name = 'report.pdf')
    
       
    pose_result = report_df.T['Pose cheating result'][0]
    obj_result = report_df.T['Num objects detected'][0]
    sound_result = report_df.T['Sound Detection result'][0]
    
    print(pose_result , obj_result ,  sound_result)   
    
        
    sesPayload = {
            
            "To": "omarhazem6@gmail.com",
            "Name": event['stud_name'],
            "path": path+'/report.pdf' ,
            "pose" : pose_result , 
            "obj" : obj_result , 
            'sound' : sound_result
            
        
    } 
    

    
    Resp = lam.invoke(FunctionName='trySES',
    InvocationType='RequestResponse',
    Payload=json.dumps(sesPayload))  
    
    
    
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(True)
    }
