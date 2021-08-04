 
import boto3
import os 
import json


lam = boto3.client('lambda')
s3 = boto3.resource('s3')
s3_c = boto3.client('s3')

def is_image (file_name) :
    supported_formats = ['png', 'jpg', 'jpeg' , 'JPG'] 
    for supported_format in supported_formats : 
        if supported_format in file_name :
            return True
    return False 

 
def is_voice (file_name) :
    supported_formats = ['WAV', 'wav','mp3','MP3'] 
    for supported_format in supported_formats : 
        if supported_format in file_name :
            return True
    return False 

def get_labels(bucket) : 
    all_labels = [] 
    object_listing = s3_c.list_objects_v2(Bucket=bucket) 
    for file in object_listing['Contents'] : 
        file_name = file["Key"]
        if is_image(file_name) or is_voice (file_name)  : 
            photo=file_name
            all_labels.append(photo)
    return all_labels
     
    
def single_file_sync (bucket , key ) :
    direct_name = '/mnt/acess/'+key.split('/')[0]
    if not os.path.isdir(direct_name) : 
        os.mkdir(direct_name)
    download_loc = '/mnt/acess/'+key
    s3.meta.client.download_file(bucket,key,download_loc)
    is_exist =  os.path.isfile(download_loc)
    print(is_exist)
    while is_exist == False :
        print('Failure in syncing , retrying process for file {}'.format(download_loc))
        s3.meta.client.download_file(bucket,key,download_loc)
    kind = 'img' if is_image(download_loc)  else 'other'
    kind = 'voice' if is_voice(download_loc) else kind
    if kind == 'img' :
        print(key.split('/')[-1])
        if key.split('/')[-1] == '1.jpeg' or key.split('/')[-1] == '1.jpg' :
            process_Image(key.split('/')[0])
    elif kind == 'voice' : 
        process_Audio(download_loc)
    print('kind' + ' ' + kind)
    print('Done '+download_loc)

def process_Image(download_loc): 
    lam.invoke(FunctionName='latest-cheating-lambda',
    InvocationType='Event',Payload=json.dumps({'path' : download_loc}) )  
    print('okokokok')
def process_Audio (download_loc) : 
    lam.invoke(FunctionName='conversion-using-ffmeg',
    InvocationType='Event',Payload=json.dumps({'file' : download_loc}) )  
    print('voice')
def sync_s3_to_efs(bucket) : 
    imgs_dir_list = []
    all_labels = get_labels(bucket)
    for image in all_labels : 
        key = image
        direct_name = '/mnt/acess/'+image.split('/')[0]
        if direct_name not in imgs_dir_list : 
            imgs_dir_list.append(direct_name)
        if not os.path.isdir(direct_name) : 
            os.mkdir(direct_name)
        download_loc = '/mnt/acess/'+image
        s3.meta.client.download_file(bucket,key,download_loc)  
    return imgs_dir_list








    



