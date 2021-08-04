import json
import os 
import subprocess
import boto3
import botocore
from sound_det import audio
import numpy as np 
import time


def create_txt_report(path , is_speech_np , detected_frames , sound_result ) : 
  vad_file = path  + "vad.txt" 
  vad_f = open(vad_file, "w") 
  vad_f.write(str(is_speech_np)+',,'+ str(detected_frames)+',,'+str(sound_result)+',,')
  vad_f.close()
  print(vad_file)



def get_array_from_txt(txt_file, split = ','):
    f = open(txt_file, "r")
    arr = f.read().split(split)
    f.close()
    return arr[:-1]
    
def getSize(filename):
    st = os.stat(filename)
    return st.st_size
def do_conversion(file ,event) : 

    mp3_file = file
    ts = time.time()
    wav_file = event['file'].replace(event['file'].split('/')[-1],'')+'{}.wav'.format(str(ts))
    subprocess.call(['/opt/ffmpeglib/ffmpeg', '-i', mp3_file,
                  wav_file])
    return wav_file
    
def parse_voice_values (txt_file) : 
    a,b,c = get_array_from_txt(txt_file,',,')
    a = [eval(i) for i in a[1:-1].split(',')]
    b = np.int64(b)
    c= eval(c)
    return a,b,c 
def lambda_handler(event, context):
    file = event['file']  
    if 'mp3' in file : 
        file = do_conversion(file,event)
    is_speech_np,detected_frames , sound_result= audio.VAD_detection(file,0.6)
    create_txt_report(event['file'].replace(event['file'].split('/')[-1],'') ,is_speech_np,detected_frames , sound_result)
    a,b,c = parse_voice_values(event['file'].replace(event['file'].split('/')[-1],'') + "vad.txt")
    print(a,b,c)
    
    # TODO implement
    #print(getSize('/tmp/stud_1.wav'))
    #s3c.upload_file('/tmp/stud_1.wav', BUCKET_NAME, 'stud_1.wav')

    return {
        'statusCode': 200,

    }
