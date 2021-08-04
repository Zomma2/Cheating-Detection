import json
import boto3
import os 
s3 = boto3.resource('s3')
ses=boto3.client('ses')
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import shutil  

def create_pdf_sendable_form (To='omarhazem6@gmail.com' , Name  = 'Student' , path = '/tmp/coff.pdf' ): 
    msg = MIMEMultipart()
    msg['Subject'] = 'Cheating Action Detected'
    msg['From'] = 'auto.grad.nodif@gmail.com'
    msg['To'] = To
    
    # what a recipient sees if they don't use an email reader
    msg.preamble = 'Multipart message.\n'
    # the message body
    part = MIMEText('''Automated Cheating Detection Detected Student : {}
    Cheating in the Exam for more information and insights please check the generated 
    pdf file below'''.format(Name))
    msg.attach(part)
    part = MIMEApplication(open(path, 'rb').read())
    part.add_header('Content-Disposition', 'attachment', filename='report.pdf')
    msg.attach(part)
    return msg 






def lambda_handler(event, context):
    
    # TODO implement
    '''
    email_address = event['Email']
    ses.verify_email_address(EmailAddress=email_address)
    '''
    '''
    ses.send_email(
        Source = 'auto.grad.nodif@gmail.com' , 
        Destination = 
        {
            'ToAddresses' : ['omarhazem6@gmail.com']
        } ,
        Message={
            'Body': {
                'Html': {
                    'Charset': CHARSET,
                    'Data': BODY_HTML,
                },
                'Text': {
                    'Charset': CHARSET,
                    'Data': BODY_TEXT,
                },
            },
            
            'Subject': {
                'Charset': CHARSET,
                'Data': SUBJECT,
            }
            }
    
    )
    '''
    
    
    
    
    
    
    '''
    bucket = 'object-detection-grad-proj'
    key= 'corr.pdf'
    s3.meta.client.download_file(bucket,key,'/tmp/coff.pdf')  
    '''
    
    # the attachment
    f = open( event['path'].replace(event['path'].split('/')[-1],'')+"/email.txt", "r")
    To = f.read()
    f.close()
    print(To)
    Name = event['Name']
    path = event['path']

    f = open( event['path'].replace(event['path'].split('/')[-1],'')+"/type.txt", "r")
    Type = f.read()
    print(Type)
    f.close()
    
    pose_result = event["pose"] 
    obj_result =event["obj"] 
    sound_result=event['sound'] 


    msg = create_pdf_sendable_form (To , Name  , path )
    


    if Type == "online" : 
        if pose_result or  obj_result or  sound_result :
            result = ses.send_raw_email(
                RawMessage =  {
                    'Data': msg.as_string() 
                    
                }
                , Source=msg['From']
                , Destinations=[msg['To']]
                )

    if Type == "file"  :
        if  obj_result or  sound_result :
            result = ses.send_raw_email(
                RawMessage =  {
                    'Data': msg.as_string() 
                    
                }
                , Source=msg['From']
                , Destinations=[msg['To']]
                )

    
    
    return {
        'statusCode': 200

    }
