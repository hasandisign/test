# change made
# change resolution 640 480
# json.loads

import math
import random  #############################################################################################################################################################
import json
import time
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import statistics
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import os
from datetime import datetime
from shapely.geometry import Point, Polygon
from imantics import Polygons, Mask
import matplotlib.pyplot as plt
from PIL import Image
import io
#from imutils import opencv2matplotlib
#from imutils.video import VideoStream
import pickle
import jetson.inference
import jetson.utils
import argparse
import sys
from threading import Thread
import boto3
from botocore.client import Config
#from imutils.video import FileVideoStream
#from imutils.video import FPS
import threading
#change here the base directory
basedir=os.environ['HOME']


if sys.version_info >= (3, 0):
	from queue import Queue
else:
	from Queue import Queue

## threading
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame)=self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped=True

def upload_video_to_s3(data1,data2,video_name,video_name2,ACCESS_KEY_ID,ACCESS_SECRET_KEY,BUCKET_NAME):
   #print("uploading cam 0 to s3")
   s3 = boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY,config=Config(signature_version='s3v4'))
   #s3.Bucket(BUCKET_NAME).put_object(Key='jetson101/camera0/'+args.file1+'.mp4', Body=data1)
   s3.Bucket(BUCKET_NAME).put_object(Key='jpn01/camera0/'+video_name, Body=data1)
   #print("uploading cam 1 to s3")
   s3 = boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY,config=Config(signature_version='s3v4'))
   #s3.Bucket(BUCKET_NAME).put_object(Key='jetson101/camera0/'+args.file1+'.mp4', Body=data1)
   s3.Bucket(BUCKET_NAME).put_object(Key='jpn01/camera1/'+video_name2, Body=data2)


## the access key and sectret key for s3 bucket along with bucket name
ACCESS_KEY_ID = 'AKIASAGFZI7HHGFXGCUR'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'jpnbuckets'


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=800, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=600, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")
is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]
try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


#load the recognition network
#net = jetson.inference.imageNet(opt.network, sys.argv)
net = jetson.inference.imageNet(argv=["--model="+basedir+"/Documents/new_classifi/jetson-inference/python/training/classification/models/ttrain_models/resnet50.onnx", "--labels="+basedir+"/Documents/new_classifi/jetson-inference/python/training/classification/data/ttrain_data/labels.txt", "--input-blob=input_0", "--output-blob=output_0",""+basedir+"/src/trt_pose/tasks/human_pose/train_scenario2.mp4"])



os.environ['MPLCONFIGDIR']="/src/trt_pose/tasks/human_pose"

myMQTTClient = AWSIoTMQTTClient("jpen01client")
myMQTTClient.configureEndpoint("a1omk7z4gjo0zp-ats.iot.us-east-2.amazonaws.com",8883)
myMQTTClient.configureCredentials(""+basedir+"/jpn01_cert/AmazonRootCA1.pem", ""+basedir+"/jpn01_cert/private.pem.key", ""+basedir+"/jpn01_cert/certificate.pem.crt")
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(10)
print("Start iot core")
myMQTTClient.connect()

# Draw the key points
def draw_keypoints(img, key):
    thickness = 3
    r=3
    w, h = img.size
    draw = PIL.ImageDraw.Draw(img)
    c=(0,255,0)  #same  color for all the  line between key points
    #kc=(255,255,0)
    #Right leg
    #draw Rankle -> RKnee (16-> 14)
    if all(key[16]) and all(key[14]):
        draw.line([ round(key[16][2] * w), round(key[16][1] * h), round(key[14][2] * w), round(key[14][1] * h)],width = thickness, fill=c)
        #print([ round(key[16][2] * w), round(key[16][1] * h)])
        draw.ellipse((round(key[16][2] * w-r), round(key[16][1] * h-r),round(key[16][2] * w+r), round(key[16][1] * h+r)), (255,0,255,0))
        #draw.arc([round(key[16][2] * w), round(key[16][1] * h), round(key[16][2] * w+5), round(key[16][1] * h+5)], 0,360, fill = kc)
    #draw RKnee -> Rhip (14-> 12)
    if all(key[14]) and all(key[12]):
        draw.line([ round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w+5), round(key[12][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[14][2] * w-r), round(key[14][1] * h-r),round(key[14][2] * w+r), round(key[14][1] * h+r)), (255,0,255,0))
    #HIP
    #draw Rhip -> Lhip (12-> 11)
    if all(key[12]) and all(key[11]):
        draw.line([ round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w+5), round(key[11][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[12][2] * w-r), round(key[12][1] * h-r),round(key[12][2] * w+r), round(key[12][1] * h+r)), (255,0,255,0))
    #left leg
    #draw Lhip -> Lknee (11-> 13)
    if all(key[11]) and all(key[13]):
        draw.line([ round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w), round(key[13][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w+5), round(key[13][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[11][2] * w-r), round(key[11][1] * h-r),round(key[11][2] * w+r), round(key[11][1] * h+r)), (255,0,255,0))
    #draw Lknee -> Lankle (13-> 15)
    if all(key[13]) and all(key[15]):
        draw.line([ round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w), round(key[15][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w+5), round(key[15][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[13][2] * w-r), round(key[13][1] * h-r),round(key[13][2] * w+r), round(key[13][1] * h+r)), (255,0,255,0))
        draw.ellipse((round(key[15][2] * w-r), round(key[15][1] * h-r),round(key[15][2] * w+r), round(key[15][1] * h+r)), (255,0,255,0))
    #RIGHT HAND
    #draw Rwrist -> Relbow (10-> 8)
    if all(key[10]) and all(key[8]):
        draw.line([ round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w), round(key[8][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w+5), round(key[8][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[10][2] * w-r), round(key[10][1] * h-r),round(key[10][2] * w+r), round(key[10][1] * h+r)), (255,0,255,0))
    #draw Relbow -> Rshoulder (8-> 6)
    if all(key[8]) and all(key[6]):
        draw.line([ round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w), round(key[6][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w+5), round(key[6][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[8][2] * w-r), round(key[8][1] * h-r),round(key[8][2] * w+r), round(key[8][1] * h+r)), (255,0,255,0))
    #SHOULDER
    #draw Rshoulder -> Lshoulder (6-> 5)
    if all(key[6]) and all(key[5]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w), round(key[5][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w+5), round(key[5][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[6][2] * w-r), round(key[6][1] * h-r),round(key[6][2] * w+r), round(key[6][1] * h+r)), (255,0,255,0))
    #LEFT HAND
    #draw Lshoulder -> Lelbow (5-> 7)
    if all(key[5]) and all(key[7]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w), round(key[7][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w+5), round(key[7][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[5][2] * w-r), round(key[5][1] * h-r),round(key[5][2] * w+r), round(key[5][1] * h+r)), (255,0,255,0))
    #draw Lelbow -> Lwrist (7-> 9)
    if all(key[7]) and all(key[9]):
        draw.line([ round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w), round(key[9][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w+5), round(key[9][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[7][2] * w-r), round(key[7][1] * h-r),round(key[7][2] * w+r), round(key[7][1] * h+r)), (255,0,255,0))
        draw.ellipse((round(key[9][2] * w-r), round(key[9][1] * h-r),round(key[9][2] * w+r), round(key[9][1] * h+r)), (255,0,255,0))
    #RIGHT LATS
    #draw Rshoulder -> RHip (6-> 12)
    if all(key[6]) and all(key[12]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w+5), round(key[12][1] * h+5)], 0,360, fill = kc)
    #LEFT LATS
    #draw Lshoulder -> LHip (5-> 11)
    if all(key[5]) and all(key[11]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w+5), round(key[11][1] * h+5)], 0,360, fill = kc)

    # HEAD
    #draw nose -> Reye (0-> 2)
    if all(key[0][1:]) and all(key[2]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w), round(key[2][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w+5), round(key[2][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[0][2] * w-r), round(key[0][1] * h-r),round(key[0][2] * w+r), round(key[0][1] * h+r)), (255,0,255,0))
    #draw Reye -> Rear (2-> 4)
    if all(key[2]) and all(key[4]):
        draw.line([ round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w), round(key[4][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w+5), round(key[4][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[2][2] * w-r), round(key[2][1] * h-r),round(key[2][2] * w+r), round(key[2][1] * h+r)), (255,0,255,0))
    #draw nose -> Leye (0-> 1)
    if all(key[0][1:]) and all(key[1]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w), round(key[1][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w+5), round(key[1][1] * h+5)], 0,360, fill = kc)
    #draw Leye -> Lear (1-> 3)
    if all(key[1]) and all(key[3]):
        draw.line([ round(key[1][2] * w), round(key[1][1] * h), round(key[3][2] * w), round(key[3][1] * h)],width = thickness, fill=c)
        draw.ellipse((round(key[1][2] * w-r), round(key[1][1] * h-r),round(key[1][2] * w+r), round(key[1][1] * h+r)), (255,0,255,0))
        draw.ellipse((round(key[3][2] * w-r), round(key[3][1] * h-r),round(key[3][2] * w+r), round(key[3][1] * h+r)), (255,0,255,0))
    # NECK
    #draw nose -> neck (0-> 17)
    if all(key[0][1:]) and all(key[17]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w), round(key[17][1] * h)],width = thickness, fill=c)
        #draw.arc([round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w+5), round(key[17][1] * h+5)], 0,360, fill = kc)
        draw.ellipse((round(key[17][2] * w-r), round(key[17][1] * h-r),round(key[17][2] * w+r), round(key[17][1] * h+r)), (255,0,255,0))
        
    #remaining circle for the points
    #cannot be done separately as image still not in frame
    #change this up
    return img

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

# fine tuning the keypoint for different classes
parser = argparse.ArgumentParser(description='TRT pose on image')
parser.add_argument('--model', type=str, default='densenet', help = 'resnet or densenet' )
parser.add_argument('--video1', type=str, default=""+basedir+"/src/trt_pose/tasks/human_pose/train_scenario2.mp4", help = 'video file name' )
parser.add_argument('--video2', type=str, default=""+basedir+"/src/trt_pose/tasks/human_pose/station_video_final.mp4", help = 'video file name' )
parser.add_argument('--image', type=str, default=""+basedir+"/src/trt_pose/tasks/human_pose/check.png", help = 'image file name' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# check if the optimized model is available already
if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet169_baseline_att_256x256_B_epoch_240.pth'
    OPTIMIZED_MODEL = 'densenet169_baseline_att_256x256_B_epoch_240_trt.pth'
    model = trt_pose.models.densenet169_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    print('-- Converting TensorRT models. This may takes several minutes...')
    model.load_state_dict(torch.load(MODEL_WEIGHTS),strict=False)
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

## for making imference on small image
def execute(img, src, t):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.circle(src, (x, y), 3, color, 2)
    print("FPS:%f "%(fps))
    draw_objects(np.asarray(img), counts, objects, peaks)
    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    out_video.write(src)
    cv2.imshow('',src)
    #return(src)

# for making inference on original image
def execute_2(img, org, count):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        #print("Human index:%d "%( i ))
        kpoint = get_keypoint(objects, i, peaks)
        org = draw_keypoints(org, kpoint)
    netfps = 1 / (end - start)  
    #draw = PIL.ImageDraw.Draw(org)
    #draw.text((459, 107), "NET FPS:%4.1f"%netfps, font=fnt, fill=(0,255,0))    
    #print("Human count:%d len:%d "%(counts[0], len(counts)))
    #print('===== Frame[%d] Net FPS :%f ====='%(count, netfps))
    return org

def get_point_from_keypoint(kp):
    actual_kp=[]
    #convert None in keypoint to 0 value and remove  first element in the tuple 
    for j in range (0,len(kp)):
        d=list(kp[j])
        del d[0]
        e=[0 if v is None else v for v in d]
        kp[j]=tuple(e)
    for i in range(0,len(kp)):
        z=(int(kp[i][1]*WIDTH* X_compress),int(kp[i][0]*HEIGHT * Y_compress))
        actual_kp.append(z)
    
    return actual_kp

count=1
fontname = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fnt = PIL.ImageFont.truetype(fontname, 24)  #set the font size 
fontScale=1
color=(255,0,0)
thickness=2

global H,W
#cap1 = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(2)
cap1=WebcamVideoStream(src=0).start() #thread for cam 1
cap2=WebcamVideoStream(src=1).start() # thread for cam 2

## 0 and 2
img = cap1.read()
H, W, __ = img.shape    # height and width of image 800*600 fow webcam
fourcc1 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
dir, filename1 = os.path.split(args.video1)
name, ext = os.path.splitext(filename1)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#out_video1 = cv2.VideoWriter(""+basedir+"src/trt_pose/tasks/human_pose/output_video/%s_%s.mp4"%(args.model, name), fourcc1, cap1.get(cv2.CAP_PROP_FPS), (W, H))
#out_video1 = cv2.VideoWriter(""+basedir+"/src/trt_pose/tasks/human_pose/%s_%s.mp4"%(args.model, name), fourcc1, 20, (W, H))
X_compress = 640 / WIDTH * 1.0     #to draw at exact key point 800.0 is the widht of source video
Y_compress = 480 / HEIGHT * 1.0    #to draw at exact key point 600.0 is the height of source video

fourcc2 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
dir, filename2 = os.path.split(args.video2)
name, ext = os.path.splitext(filename2)
#out_video2 = cv2.VideoWriter('/home/bipun/src/trt_pose/tasks/human_pose/output_video/%s_%s.mp4'%(args.model, name), fourcc2, cap2.get(cv2.CAP_PROP_FPS), (W, H))
#out_video2 = cv2.VideoWriter(""+basedir+"/src/trt_pose/tasks/human_pose/output_video/%s_%s.mp4"%(args.model, name), fourcc2, cap2.get(cv2.CAP_PROP_FPS), (W, H))
X_compress = 640 / WIDTH * 1.0     #to draw at exact key point 800.0 is the widht of source video
Y_compress = 480 / HEIGHT * 1.0    #to draw at exact key point 600.0 is the height of source video
if cap1 and cap2 is None:
    print("Camera Open Error")
    sys.exit(0)
parse_objects = ParseObjects(topology, cmap_threshold = 0.05, link_threshold= 0.05)  # default 0.1 and 0.1 good at 0.05   0.05 ########################################################################
draw_objects = DrawObjects(topology)
fnt = PIL.ImageFont.truetype(fontname, 24)  #set the font size 
lnF=1012*762*3

# Train classification
#input = jetson.utils.videoSource('/dev/video0', argv=sys.argv)   #/dev/video0 for camera1
#output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()
capture_time=10
start_time=time.time() ## record time
out=None
random_hour = random.sample(range(5,24),3) ######################################################################################################################################################
#random_hour = [5,7,10,12,15,17,20,23]
#print(random_hour)
#manual labeling polygons
c1fr =open("cam1redzone.txt","r")

cam1_red=json.loads(c1fr.read())

print(cam1_red)
c1fy =open("cam1yellowzone.txt","r")
cam1_yellow=json.loads(c1fy.read())
c1fg = open("cam1greenzone.txt","r")
cam1_green=json.loads(c1fg.read())
c2fr = open("cam2redzone.txt","r")
cam2_red=json.loads(c2fr.read())
c2fy = open("cam2yellowzone.txt","r")
cam2_yellow=json.loads(c2fy.read())
c2fg = open("cam2greenzone.txt","r")
cam2_green=json.loads(c2fg.read())
c1fr.close()
c1fy.close()
c1fg.close()

c2fr.close()
c2fy.close()
c2fg.close()


cam1_red.append(cam1_red[0])

pol_cam1_red=Polygon(cam1_red)
cam1_yellow.append(cam1_yellow[0])
pol_cam1_yellow=Polygon(cam1_yellow)
cam1_green.append(cam1_green[0])
pol_cam1_green=Polygon(cam1_green)
cam2_red.append(cam2_red[0])
pol_cam2_red=Polygon(cam2_red)
cam2_yellow.append(cam2_yellow[0])
pol_cam2_yellow=Polygon(cam2_yellow)
cam2_green.append(cam2_green[0])
pol_cam2_green=Polygon(cam2_green)
while True:
    #print(cam1_red)
    now = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    now_upload = datetime.now()
    timestamp = datetime.timestamp(now_upload)
    #print(timestamp)
    if now_upload.hour==5:
        random_hour = random.sample(range(5,24),3)    
    #print(now_upload.hour in random_hour)
    dst1 = cap1.read()
    #ret_val2, dst2 = cap2.read()
    dst2 = cap2.read()
    fname="Cam0_"+now
    fname2="Cam1_"+now
    full_fname="Cam0_"+now+".mp4"
    full_fname2="Cam1_"+now+".mp4"
    
    if  now_upload.hour in random_hour and now_upload.minute==0 and now_upload.second==0:
        out=cv2.VideoWriter(full_fname, fourcc, 5.0, (640,480))  #record video
        video_name=full_fname   ### name at 00 seconds used down to upload file with this name
        out2=cv2.VideoWriter(full_fname2, fourcc, 5.0, (640,480))  #record video
        video_name2=full_fname2   ### name at 00 seconds used down to upload file with this name
    #print(now)
    if  now_upload.minute<=capture_time and now_upload.hour in random_hour and out!= None:
        #print("recording")
        out.write(dst1)
        #cv2.imshow('cam1 record',dst1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        out2.write(dst2)
        #cv2.imshow('cam2 record',dst2)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
    if now_upload.minute>capture_time and now_upload.hour in random_hour and out!= None:
        out.release()
        out2.release()
        data1 = open(video_name, 'rb')
        data2 = open(video_name2, 'rb')
        #print(args.file1)
        #uploading to s3 bucket
        #th = threading.Thread(target=upload_video_to_s3, args=(data1,data2,video_name,video_name2,ACCESS_KEY_ID,ACCESS_SECRET_KEY,BUCKET_NAME ))
        #th.start()
        
        out=None
        out2=None
        #If out and out2 are both none delete the video files as they signify complete upload
        #get he duration of video 
########os.system('rm '+video_name+' '+video_name2)
        
        #path_upload=full_fname
        #print(path_upload)
        #if os.path.exists(path_upload) and now_upload.second>10: #upload video only if file exits and check if complete
            #data2 = open(full_fname+'.mp4', 'rb')
        #th.join()
# Release everything if job is finished
    #if ret_val1 and ret_val2 == False:
     #   print("Frame Read End")
      #  break
    ######trin class
    
    img11 = cv2.resize(dst1, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    cuda_mem1 = jetson.utils.cudaFromNumpy(img11) # image to cuda imag opencv image does not work as it is array
    #imgg = input.Capture()
    class_id1, confidence1 = net.Classify(cuda_mem1)
    print("camera1:")
    print(class_id1)
    img22 = cv2.resize(dst2, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    cuda_mem2 = jetson.utils.cudaFromNumpy(img22) # image to cuda imag opencv image does not work as it is array
    #imgg = input.Capture()
    class_id2, confidence2 = net.Classify(cuda_mem2)
    print("camera2:")
    print(class_id2)
    #####train class
    if class_id1==0 or class_id1==1:
        img1 = cv2.resize(dst1, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        pilimg1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB)
        pilimg1 = PIL.Image.fromarray(pilimg1)
        pilimg1 = execute_2(img1, pilimg1, count)  #for execute 2
        array1 = np.asarray(pilimg1, dtype="uint8")
        final_img1=cv2.cvtColor(array1,cv2.COLOR_RGB2BGR)   #CONVERT THE IMAGE ORIGINAL COLOR i.e BGR
        if cv2.waitKey(1) & 0xFF ==ord('q'):    
            break
        data1 = preprocess(img1)
        cmap1, paf1 = model_trt(data1)
        cmap1, paf1 = cmap1.detach().cpu(), paf1.detach().cpu()
        counts1, objects1, peaks1 = parse_objects(cmap1, paf1)
        draw1 = PIL.ImageDraw.Draw(pilimg1)
        for i in range(counts1[0]):
        #print("Human index:%d "%( i ))
            kpoint1 = get_keypoint(objects1, i, peaks1)
            z1=get_point_from_keypoint(kpoint1)
        
        
            la_151=z1[15]
            ra_161=z1[16]
        
            #cv2.putText(final_img1,"left ankle",la_151,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #cv2.putText(final_img1,"right ankle",ra_161,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

       
        #draw.text(ra_16,"Right ankle", font=fnt,fill=(0,255,0))
            #for k in range(0, len(new_original_roi1)):
            
            #ground_truth_binary_mask1 = np.array(temp_mask1[k,:,:], dtype = np.uint8)
            #polygons1 = Mask(ground_truth_binary_mask1).polygons()
            
            #coord1 = (polygons1.points[0])
            #coord1 = (coord1.tolist())
            #coord1.append(coord1[0])
            la_x1=la_151[0]
            la_y1=la_151[1]
            ra_x1=ra_161[0]
            ra_y1=ra_161[1]
            pp1=Point(la_x1,la_y1)
            #poly1=Polygon(coord1)
            pp1=Point(la_x1,la_y1)
            if pol_cam1_red.contains(pp1):
                cv2.putText(final_img1,"Red Alert",(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Red alert")
                print("publishing message from jetson")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera001", "timestamp": '+ str(timestamp) +', "message": "RED ALERT"}')
            if pol_cam1_yellow.contains(pp1):
                cv2.putText(final_img1,"Yellow Aalert",(50,120),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Yellow alert")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera001", "timestamp": '+ str(timestamp) +', "message": "YELLOW ALERT"}')
            if pol_cam1_green.contains(pp1):
                cv2.putText(final_img1,"Green Alert",(50,140),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Ggreeen alert")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera001", "timestamp": '+ str(timestamp) +', "message": "GREEN ALERT"}')
        cv2.imshow("cam 1",final_img1)
    # camp 2
    if class_id2==0 or class_id2==1:
        img2 = cv2.resize(dst2, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        pilimg2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        pilimg2 = PIL.Image.fromarray(pilimg2)
        pilimg2 = execute_2(img2, pilimg2, count)  #for execute 2
        array2 = np.asarray(pilimg2, dtype="uint8")
        final_img2=cv2.cvtColor(array2,cv2.COLOR_RGB2BGR)   #CONVERT THE IMAGE ORIGINAL COLOR i.e BGR
        if cv2.waitKey(1) & 0xFF ==ord('q'):    
            break
        data2 = preprocess(img2)
        cmap2, paf2 = model_trt(data2)
        cmap2, paf2 = cmap2.detach().cpu(), paf2.detach().cpu()
        counts2, objects2, peaks2 = parse_objects(cmap2, paf2)
        draw2 = PIL.ImageDraw.Draw(pilimg2)
        for j in range(counts2[0]):
        #print("Human index:%d "%( i ))
            kpoint2 = get_keypoint(objects2, j, peaks2)
            z2=get_point_from_keypoint(kpoint2)
        
        
            la_152=z2[15]
            ra_162=z2[16]
        
            #cv2.putText(final_img2,"left ankle",la_152,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #cv2.putText(final_img2,"right ankle",ra_162,cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    

            
            #ground_truth_binary_mask2 = np.array(temp_mask2[k,:,:], dtype = np.uint8)
            #polygons2 = Mask(ground_truth_binary_mask2).polygons()
            #coord2 = (polygons2.points[0])
            #coord2 = (coord2.tolist())
            #coord2.append(coord2[0])
            la_x2=la_152[0]
            la_y2=la_152[1]
            ra_x2=ra_162[0]
            ra_y2=ra_162[1]
            pp2=Point(la_x2,la_y2)
            #poly2=Polygon(coord2)
            pp2=Point(la_x2,la_y2)
            if pol_cam2_red.contains(pp2):
                cv2.putText(final_img2,"Red Alert",(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Red alert")
                print("publishing message from jetson")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera002", "timestamp": '+ str(timestamp) +', "message": "RED ALERT"}')
            if pol_cam2_red.contains(pp2):
                cv2.putText(final_img2,"Yellow Alert",(50,120),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Yellow alert")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera002", "timestamp": '+ str(timestamp) +', "message": "YELLOW ALERT"}')
            if pol_cam2_red.contains(pp2):
                cv2.putText(final_img2,"Green Alert",(50,140),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            #print("Ggreeen alert")
                myMQTTClient.publish(topic = "home/zones",QoS=1,payload='{"cameraID": "camera002", "timestamp": '+ str(timestamp) +', "message": "GREEN ALERT"}')
        cv2.imshow("cam 2",final_img2)
        #finale=cv2.hconcat([final_img1,final_img2])
        #cv2.imshow("feed",finale)
        
    #is_success, im_buf_arr = cv2.imencode(".jpg", img1)
    #byte_im = im_buf_arr.tobytes()
    #byte_im.shape
    #MQTT_MESSAGE = dst1.tostring()
    #mqttc.publish(MQTT_PATH, MQTT_MESSAGE)
    #np_arr_rgb=opencv2matplotlib(dst1[1:500,1:500,:])
    #image= Image.fromarray(np_arr_rgb)
    #imgbytarray=io.BytesIO()
    #image.save(imgbytarray,'PNG')
    #vidd=imgbytarray.getvalue()
    #vidd=json.dumps(vid) 
    #vid=dst1.reshape(2313432,1)
    ####print(vid.shape)
    #vide=vid[0:10].tolist()
    #vide=[1,2,3,4]
    #print(vide[0:5])
    #print(vide[1:10])
    #vidd=json.dumps(vide)
    #print(type(vidd))
    #myMQTTClient.publish(topic = "home/zones",QoS=1,payload=vidd)
    #finale=cv2.hconcat([final_img1,final_img2])
    #cv2.imshow('asdf',finale)
    #if cv2.waitKey(1) & 0xFF ==ord('q'):    
    #        break
    #out_video1.write(final_img2)       # THIS ONLY TO SAVE THE final inferred VIDEO
    
#client.loop_stop()
#client.disconnect()

if out!= None:
    out.release()
if out2!= None:
    out2.release()
#out_video1.release()
cap1.stop()
cap2.stop()
cv2.destroyAllWindows()

