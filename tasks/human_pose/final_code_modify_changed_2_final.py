import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
import math
import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import os
import time
from datetime import datetime
import statistics
os.environ['MPLCONFIGDIR']="/src/trt_pose/tasks/human_pose"

# The first parameter is angle 
def angle_calc(p0, p1, p2 ):
    '''
        p1 is center point from where we measured angle between p0 and
    '''
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

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
parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='densenet', help = 'resnet or densenet' )
#parser.add_argument('--video', type=str, default='/home/ati/src/trt_pose/tasks/human_pose/danger_len_1.mp4', help = 'video file name' )
parser.add_argument('--video', type=str, default='/dev/video0', help = 'video file name' )
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
''' 
Draw to original image
'''
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

def angle_cal(p0,p1,p2):
    try:
        a =  (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b =  (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c =  (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angie = math.acos((a+b-c)/math.sqrt(4*a*b)) * 180/math.pi
    except:
        return 0
    return int(angie)


def calc_angles(ls_5,rs_6,le_7,re_8,lw_9,rw_10,lh_11,rh_12,lk_13,rk_14,la_15,ra_16,h_0):
    ang_0511=angle_cal(h_0,ls_5,lh_11)
    ang_579 = angle_cal(ls_5,le_7,lw_9)
    ang_51113 = angle_cal(ls_5,lh_11,lk_13)
    ang_111315 = angle_cal(lh_11,lk_13,la_15)
    #ang_6810 = angle_cal()
    ang_0612=angle_cal(h_0,rs_6,rh_12)
    ang_6810 = angle_cal(rs_6,re_8,rw_10)
    ang_61214 = angle_cal(rs_6,rh_12,rk_14)
    ang_121416 = angle_cal(rh_12,rk_14,ra_16)

    all_angle = [ang_0511, ang_579, ang_51113,ang_111315,ang_0612, ang_6810, ang_61214, ang_121416]

    return all_angle

def euclidian(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

#def normal_walk_check(param):
def alert_frame_start(pilimg):
    frames=1
    
    if frames in range(1,50):
    #if alert_frame in range(1,50):
        draw = PIL.ImageDraw.Draw(pilimg)
        draw.text((100, 70), "ALERT", font=fnt, fill=(0,0,255))

global H,W
cap = cv2.VideoCapture(args.video)
ret_val, img = cap.read()
H, W, __ = img.shape    # heihty and width of image 800*600
print(W)
print(H)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
dir, filename = os.path.split(args.video)
name, ext = os.path.splitext(filename)
out_video = cv2.VideoWriter('./%s_%s.mp4'%( args.model, name), fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))
count = 0
### Change here if the input pixel size is diffent eg if 1920 × 1080
# X_compress = 1920 /WIDTH * 1.0
# Y_compress =1080 /HEIGHT * 1.0
X_compress = 1920 / WIDTH * 1.0     #to draw at exact key point 800.0 is the widht of source video
Y_compress = 1080 / HEIGHT * 1.0    #to draw at exact key point 600.0 is the height of source video

if cap is None:
    print("Camera Open Error")
    sys.exit(0)
parse_objects = ParseObjects(topology, cmap_threshold = 0.1, link_threshold= 0.1)  # default 0.1 and 0.1 good at 0.05   0.05
draw_objects = DrawObjects(topology)
fontname = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fnt = PIL.ImageFont.truetype(fontname, 24)  #set the font size 

global start_sit, start_sit_time, end_sit_time, first_decision_time, all_key_frame_count_stationary,all_key_frame_count_move
start_first_decision_time_stationary = datetime.now()
end_first_decision_time_stationary = datetime.now()
start_first_decision_time_stationary_stand = datetime.now()
end_first_decision_time_stationary_stand = datetime.now()
start_first_decision_time_move = datetime.now()
end_first_decision_time_move = datetime.now()
start_sit= 0
first_decision_time_stationary = 0
first_decision_time_stationary_stand = 0

first_decision_time_move = 0

test0=[]
all_key_frame_count_stationary=0
all_key_frame_count_stationary_stand=0
all_key_frame_count_move=0
move1=[]
move2=[]
count = 1
frame=0
current_frame=-1
stop_counter_stationary= False

alert_value_stationary=False
alert_value_stand= False
alert_value_move=False
while True:
    
    frame=frame+1
    
    ret_val, dst = cap.read()
    
    if ret_val == False:
        print("Frame Read End")
        break
    #out_video.write(final_img) # This is to save the raw video
    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    pilimg = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    pilimg = PIL.Image.fromarray(pilimg)
    if alert_value_stationary== True:
        draw = PIL.ImageDraw.Draw(pilimg)
        draw.text((10, 95), "ALERT SITTING", font=fnt, fill=(255,0,0))
    if alert_value_stand== True:
        draw = PIL.ImageDraw.Draw(pilimg)
        draw.text((10, 145), "ALERT STANDING", font=fnt, fill=(255,0,0))
    
    if alert_value_move== True:
        draw = PIL.ImageDraw.Draw(pilimg)
        draw.text((10, 220), "ALERT Walking", font=fnt, fill=(255,0,0)) 
    #pilimg = execute(img, np.asarray(pilimg), count) # for execture
    pilimg = execute_2(img, pilimg, count)  #for execute 2
    array = np.asarray(pilimg, dtype="uint8")
    final_img=cv2.cvtColor(array,cv2.COLOR_RGB2BGR)   #CONVERT THE IMAGE ORIGINAL COLOR i.e BGR
    
    count += 1
    if cv2.waitKey(1) & 0xFF ==ord('q'):    
        break

    #Classification of people in video make decision after 3 seconds has passed normal/abnormal
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    #cv2.putText(final_img,"Frame:%s"%count,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    alert_frame_start(pilimg) 
    draw = PIL.ImageDraw.Draw(pilimg)
    draw.text((100, 70), "ALERT", font=fnt, fill=(0,0,255))
    #j1=[]
    for i in range(counts[0]):
        #print("Human index:%d "%( i ))
        kpoint = get_keypoint(objects, i, peaks)
        #check_full_keypoint=[item for item in kpoint if kpoint[0] == None or kpoint[1] == None or kpoint[2] == None] #for each objects in a frame check if all key points exits
        z=get_point_from_keypoint(kpoint)
        
        #end_first_decision_time = datetime.now()
        if z.count((0,0)) <= 4: # process if less than 2 key points missing For occluded images
            #start timer here
            #start_first_decision_time = datetime.now()
            #end_first_decision_time = datetime.now()
            h_0=z[0]
            n_17=z[17]
            ls_5=z[5]
            rs_6=z[6]
            le_7=z[7]
            re_8=z[8]
            lw_9=z[9]
            rw_10=z[10]
            lh_11=z[11]
            rh_12=z[12]
            lk_13=z[13]
            rk_14=z[14]
            la_15=z[15]
            ra_16=z[16]
            
            
            all_angles = calc_angles(ls_5,rs_6,le_7,re_8,lw_9,rw_10,lh_11,rh_12,lk_13,rk_14,la_15,ra_16,h_0)
            
            move1.append(lh_11[0])
            move2.append(lh_11[1])
            #print(move[current_frame])
            current_frame=current_frame+1
            if all_key_frame_count_stationary ==1:
                start_first_decision_time_stationary = datetime.now()
            if all_key_frame_count_stationary >=2 :   
                end_first_decision_time_stationary = datetime.now()
                
            first_decision_time_stationary = (end_first_decision_time_stationary-start_first_decision_time_stationary).total_seconds()
            
            if all_key_frame_count_stationary_stand==1:
                start_first_decision_time_stationary_stand = datetime.now() 
            if all_key_frame_count_stationary_stand >=2 :   
                end_first_decision_time_stationary_stand = datetime.now() 
            first_decision_time_stationary_stand = (end_first_decision_time_stationary_stand-start_first_decision_time_stationary_stand).total_seconds()


            if all_key_frame_count_move ==1:
                start_first_decision_time_move = datetime.now()
            if all_key_frame_count_move >=2 : 
                end_first_decision_time_move = datetime.now()
            first_decision_time_move = (end_first_decision_time_move-start_first_decision_time_move).total_seconds()
            
            #print(first_decision_time_move)
            #print(first_decision_time_stationary)
            #print(first_decision_time_move)
            if current_frame>=3: 
               
                #print(abs(move1[current_frame]-move1[current_frame-5]))    
                if (abs(move1[current_frame]-move1[current_frame-3]) > 3 and abs(move2[current_frame]-move2[current_frame-3]) > 3) :
                    #j1.append(False)
                    all_key_frame_count_move+=1
                    cv2.putText(final_img,"moving",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) 
                    if first_decision_time_move >= 2:
                        if (all_angles[3] in range(120,180) and all_angles[7] in range(120,180)) and all_angles[2] in range(120,180) and all_angles[6] in range (120,180):
                            cv2.putText(final_img,"walking",(10,195),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) #To print text at keypoints 
                           
                        elif (all_angles[2] not in range(115,170) and all_angles[6] not in range (115,170) and all_angles[3] not in range(115,180) and all_angles[7] not in range(115,180)) or lw_9[1]>ls_5[1] or rw_10[1] > rs_6[1] :
                            #cv2.putText(final_img,"Danger",(10,135),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2) 
                            #print(all_angles[2])
                            #print(all_angles[3])
                            alert_value_move= True
                            
                        else:  
                            cv2.putText(final_img,"Walking",(10,195),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) 
                            #alert_value_move= True
                            
                        
                    if first_decision_time_move >= 5:
                        all_key_frame_count_move = 1
                        alert_value_move=False
                       
                #else:
                    #j1.append(True)
    
                if (abs(move1[current_frame]-move1[current_frame-3]) < 3 and abs(move2[current_frame]-move2[current_frame-3]) < 3) :
                    #j1.append(True)
                    all_key_frame_count_stationary+=1
                    
                    if (first_decision_time_stationary>=2 or all_key_frame_count_stationary_stand>=2):
                     
                    #all_key_frame_count_move = 0 
                    
                        cv2.putText(final_img,"stationary",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) 
                    
                        dist1 = euclidian(lh_11,la_15)
                        dist2 = euclidian(lk_13,la_15)
                        dist3 = euclidian(ls_5,lh_11)
                    
                    try:
                    
                        ratio_dist2=(dist1/dist3)
                    except:
                        pass
                    
                    
                    if (all_angles[3] in range(70,140) and all_angles[6] in range(70,140)) or ((all_angles[2] in range(60,170) and all_angles[7] in range(60,160) )):  #side pose sittiing and ratio_dist2 < 1.1)
                        

                        if first_decision_time_stationary >= 4 :
                            start_sit =start_sit + 1
                            cv2.putText(final_img,"sitting",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) 
                            
                            if first_decision_time_stationary >= 6 and first_decision_time_stationary <= 10:
                                cv2.putText(final_img,"Normal",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                                #alert_frame_start(pilimg)
                            if first_decision_time_stationary >= 10:
                                #if frames in range(1,50):
                                #if alert_frame in range(1,50):
                                alert_value_stationary=True    
                                cv2.putText(final_img,"Danger",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
                                
                            if first_decision_time_stationary >= 13:
                                all_key_frame_count_stationary=1
                                alert_value_stationary=False
                                start_sit=0
                            
                        
                    if (all_angles[2] in range (170,180) and all_angles[6] in range(170,180) and all_angles[3] in range(165,180) and all_angles[7] in range (160,180)): 
                        all_key_frame_count_stationary_stand+=1
                        if first_decision_time_stationary_stand>= 4:   
                            if (all_angles[3] in range(165,180) and all_angles[7] in range(165,180)) and all_angles[2] in range(170,180) or all_angles[6] in range (170,180):
                                cv2.putText(final_img,"standing-NORMAL",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                                #cv2.putText(final_img,"Normal",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                                #start_sit = 0
                                
                            else:
                                #cv2.putText(final_img,"Danger",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                alert_value_stand= True
                                #start_sit = 0
                            if first_decision_time_stationary_stand>= 8:
                                all_key_frame_count_stationary_stand=1  
                #else:
                   #j1.append(False)
                
            #all_key_frame_count_stationary+=1
            #all_key_frame_count_move+=1
    #for j in range(counts[0]):
    #print(j1)    
    #if all([x == False for x in j1]):
     #   all_key_frame_count_stationary = 1
    

    cv2.imshow("OUTPUT_VIDEO",final_img) 
    out_video.write(final_img)       # THIS ONLY TO SAVE THE final inferred VIDEO
cv2.destroyAllWindows()
out_video.release()
cap.release()
