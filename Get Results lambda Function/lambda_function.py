import json
import os 
import time 
import requests 
import boto3
from os import path as Path
lam = boto3.client('lambda')

def get_array_from_txt(txt_file,split = ','):
    f = open(txt_file, "r")
    arr = f.read().split(split)
    f.close()
    return arr[:-1]
    
def run(path , event) : 
    cheating_result = None
    yolo_result  = None
    sound_result = None
    try :
        cheating_result , yolo_result , sound_result = [eval(i) for i in get_array_from_txt(path)]
    except : 
        time.sleep(0.7)
        try :
            cheating_result , yolo_result , sound_result = [eval(i) for i in get_array_from_txt(path)]
        except :
            time.sleep(0.4)
            try : 
                cheating_result , yolo_result , sound_result = [eval(i) for i in get_array_from_txt(path)]
            except : 
                time.sleep(0.4)
                try : 
                    cheating_result , yolo_result , sound_result = [eval(i) for i in get_array_from_txt(path)]
                except : 
                    time.sleep(0.8)
                    try : 
                        cheating_result , yolo_result , sound_result = [eval(i) for i in get_array_from_txt(path)]
                    except :
                        print('No Files to parse critical error')
            
    
        
    try :
        os.remove(path)
    except :
        pass
    print(cheating_result , yolo_result , sound_result)
    if eval(event['body'])['questionsType'] == "online" : 
        if cheating_result or  yolo_result or  sound_result :
            print('online')
            '''
            if result == True :
                x = requests.post('http://e4ab892515ea.ngrok.io/cheatingDetection/getResult', json = {'userId':'6071299a5bf57641c44f84bd',
                'examId' : '1' , 'username' : '2'
                })
            print(x)
            '''
        
    if eval(event['body'])['questionsType'] == "file"  :
        if  yolo_result or  sound_result :
            print('file')
            '''
            if result == True :
                x = requests.post('http://e4ab892515ea.ngrok.io/cheatingDetection/getResult', json = {'userId':'6071299a5bf57641c44f84bd',
                'examId' : '1' , 'username' : '2'
                })
            print(x)
            '''
    print(path)

    
    
        
'''
def start (path , delay) :
    try : 
        run(path)
    except : 
        time.sleep(delay)
        try :
            run(path)
        except : 
            time.sleep(delay+1)
            try :
                run(path)
            except :
                time.sleep(delay+2)
                try : 
                    run(path)
                except :
                    print('Error in parsing no results to parse , please double check result writing functions')
        
'''    
def lambda_handler(event, context):
    # TODO implement

    email = eval(event['body'])['InstructorEmail'][0]
    type_of_exam =  eval(event['body'])['examType']
    
    f = open('/mnt/acess/'+eval(event['body'])['path'] +"/email.txt", "w+")
    f.write(email)
    f.close()
    
    
    f = open('/mnt/acess/'+eval(event['body'])['path'] +"/type.txt", "w+")
    f.write(type_of_exam)
    f.close()
    


    
    
    return {
        'statusCode': 200
    }
