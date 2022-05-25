import numpy as np
import cv2
import time
import os
import datetime
# Define the duration (in seconds) of the video capture here
capture_duration = 10

cap = cv2.VideoCapture(0)
now = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
now_upload = datetime.datetime.now()
fname="Cam0_"+now
full_fname="Cam0_"+now+".mp4"
print(fname)
print(now_upload.second)
#+now.day+ now.hour+now.minute+now.second)
#now1=$(date +"cam0_%m_%d_%Y_%H_%M_%S")
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out=None
print(out !=None)
out=cv2.VideoWriter(full_fname, fourcc, 20.0, (640,480))


start_time = time.time()
while( int(time.time() - start_time) < capture_duration ):
    now_upload = datetime.datetime.now()
    print(now_upload.second)
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
command="python3 upload_video_final.py --file2="+fname
print(command)


# Release everything if job is finished

out.release()
#os.system(command)  #uncomment to upload video to s3
cap.release()
cv2.destroyAllWindows()
