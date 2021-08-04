from image_ser import serializer
import boto3
import json
import numpy as np 

runtime = boto3.Session().client('sagemaker-runtime')


def predict(image,endpointName) :
    image = serializer._npy_dumps(image)
    response = runtime.invoke_endpoint(EndpointName=endpointName,
                                   ContentType='application/x-npy',
                                   Body= image )   
    for x in response['Body'].iter_lines() : 
        response = x
    response = json.loads(response) 
    response = np.array(response)
    return response

