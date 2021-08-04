import json
import boto3

ses=boto3.client('ses')

def lambda_handler(event, context):
    # TODO implement
    
    email_address = eval(event['body'])['InstructorEmail']
    ses.verify_email_address(EmailAddress=email_address)
    return {
        'statusCode': 200
    }
