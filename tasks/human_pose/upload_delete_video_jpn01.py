#! /usr/bin/python
import os
import boto3
from botocore.client import Config

ACCESS_KEY_ID = 'AKIASAGFZI7HHGFXGCUR'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'jpnbuckets'

path = "/home/disign/src/trt_pose/tasks/human_pose/"
video_camera0 = []
video_camera1 = []

for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'Cam0' in i:
        video_camera0.append(i)
    if os.path.isfile(os.path.join(path,i)) and 'Cam1' in i:
        video_camera1.append(i)
#print(video_camera0)
#print(video_camera1)
for files in video_camera0:
    #print(files)
    video_name=files
    print(video_name)
    data1 = open(video_name, 'rb')
    s3 = boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY,config=Config(signature_version='s3v4'))
    s3.Bucket(BUCKET_NAME).put_object(Key='jpn01/camera0/'+video_name, Body=data1)
for files in video_camera1:    
    video_name2=files
    data2 = open(video_name2, 'rb')
    s3 = boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY,config=Config(signature_version='s3v4'))
    s3.Bucket(BUCKET_NAME).put_object(Key='jpn01/camera1/'+video_name2, Body=data2)
for files in video_camera0:
    video_name=files
    os.system('rm '+video_name)
for files in video_camera1:
    video_name2=files
    os.system('rm '+video_name2)
