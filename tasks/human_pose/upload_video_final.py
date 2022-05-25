import boto3
from botocore.client import Config
import argparse

parser = argparse.ArgumentParser(description='Upload file to S3')
#parser.add_argument('--file1', type=str, default="", help = 'camera 0 video' )
parser.add_argument('--file2', type=str, default="", help = 'camera 1 video' )
args = parser.parse_args()
ACCESS_KEY_ID = 'AKIAYPSBHU55JYG4MQU7'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'jetson101bucket'

#data1 = open(args.file1+'.mp4', 'rb')
data2 = open(args.file2+'.mp4', 'rb')
#print(args.file1)
print(args.file2)
s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)
#s3.Bucket(BUCKET_NAME).put_object(Key='jetson101/camera0/'+args.file1+'.mp4', Body=data1)
s3.Bucket(BUCKET_NAME).put_object(Key='jetson101/camera1/'+args.file2+'.mp4', Body=data2)
#s3.Bucket(BUCKET_NAME).put_object(Key='/jetson101/camera0/05_17_2021_13_04.mp4', Body=data)

print ("Finished Uploading")
