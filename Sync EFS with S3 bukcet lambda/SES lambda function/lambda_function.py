import json
import urllib.parse
import boto3
from Sync import s3_efs
lam = boto3.client('lambda')
print('Loading function')
s3 = boto3.client('s3')


def lambda_handler(event, context):
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    
    
    s3_efs.single_file_sync(bucket , key )

    return {
        'Status' : 200
    }