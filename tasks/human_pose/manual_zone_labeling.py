# dated: 29 july,2021
import json
import numpy as np
#from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
from imantics import Polygons, Mask
from shapely.geometry import Point, Polygon
root = Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Japan Station Monitoring')
frame.config(background='light blue')
label = Label(frame, text="Japan Station Monitoring", bg='light blue', font=('Times 30 bold'))
label.pack(side=TOP)
#filename = PhotoImage(file="/home/bipun/Pictures/check_3.png")
#background_label = Label(frame, image=filename)
#background_label.pack(side=TOP)

def hel():
    help(cv2)


def Contri():
    tkinter.messagebox.showinfo("Contributors", "\n1. Hiroyuki Miyazaki\n2. Bipun Man Pati \n")


def anotherWin():
    tkinter.messagebox.showinfo("About",
                                'Japan Station Monitoring version v1.0\n Made for manual\n-labeling of\n-Red zone\n-Green Zone and\n Green Zone')


menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools", menu=subm1)
subm1.add_command(label="Open CV Docs", command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About", menu=subm2)
subm2.add_command(label="Japan Station Monitoring", command=anotherWin)
subm2.add_command(label="Contributors", command=Contri)


def left_click_detect(event, x, y, flags, points):
    if (event == cv2.EVENT_LBUTTONDOWN):
        #print(f"\tClick on {x}, {y}")
        points.append([x,y])
        #print(points)

        current_point = points
    if event == cv2.EVENT_MBUTTONDOWN:
        print("points captured")
        #f = open("redzone.txt", "w")
        #f.write(json.dumps(points ))

        #f.close()
        #points.clear()


def exitt():
    exit()


# used to record the time when we processed last frame

def web1():
    capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("/home/bipun/Downloads/Cam1_2021_07_25-12_00_00_PM.mp4")  # Open video file
    #capture.set(cv2.CV_CAP_PROP_FPS, 5)

    #while True:
    while capture.read():
        capture.set(cv2.CAP_PROP_FPS, 0.1)
        ret, frame = capture.read()
        if not ret:
            break
       #print(frame.shape)
       #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)
        if cv2.waitKey(200) & 0xFF == ord('q'): #1000/200=5 fps
            break
    capture.release()
    cv2.destroyAllWindows()


def web2():
    capture = cv2.VideoCapture(1)
    while True:
        ret, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def redzonecam1(*args):
    print('Camera1 Red Zone saved',args)
    #cv2.displayOverlay('frame', "Overlay 5secs", 5000);
    with open('cam1redzone.txt', 'w') as f:
        f.write(json.dumps(points))

    f.close()
    f = open("cam1redzone.txt", "r")
    list_of_lists=json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol=Polygon(list_of_lists)
    p1=Point(10,20)
    print(pol.contains(p1))
    f.close()
    points.clear()

def yellowzonecam1(*args):
    print('Camera1 Yellow Zone saved',args)
    with open('cam1yellowzone.txt', 'w') as f:
        f.write(json.dumps(points))
    f.close()
    f = open("cam1yellowzone.txt", "r")
    list_of_lists = json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol = Polygon(list_of_lists)
    p1 = Point(10, 20)
    print(pol.contains(p1))
    f.close()
    points.clear()


def greenzonecam1(*args):
    print('Camera1 Green Zone saved',args)
    with open('cam1greenzone.txt', 'w') as f:
        f.write(json.dumps(points))
    f.close()
    f = open("cam1greenzone.txt", "r")
    list_of_lists = json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol = Polygon(list_of_lists)
    p1 = Point(10, 20)
    print(pol.contains(p1))
    f.close()
    points.clear()


def draw_poly_cam1():
    #CV_GUI_NORMAL = 0x10
    capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture("/home/bipun/Downloads/Cam1_2021_07_25-12_00_00_PM.mp4")  # Open video file
    polygon = []
    global points
    points = []
    cv2.namedWindow('Frame',cv2.WINDOW_AUTOSIZE)
    cv2.createButton("RED ZONE", redzonecam1, None, cv2.QT_PUSH_BUTTON, 1)
    cv2.createButton("YELLOW ZONE", yellowzonecam1, None, cv2.QT_PUSH_BUTTON, 1)
    cv2.createButton("GREEN ZONE", greenzonecam1, None, cv2.QT_PUSH_BUTTON, 1)
    while (capture.isOpened()):
        ret, frame = capture.read()  # read a frame
        if not ret:
            print('EOF')
            break
        cv2.polylines(frame, np.array([points]), False, (255,255,0), 3)
        #frame = cv2.polylines(frame, polygon, False, (255, 0, 0), thickness=5)
        i = len(points)
        #if i > 1:
            #x1 = points[i - 1][0]
            #y1 = points[i - 1][1]
            #x2 = points[i - 2][0]
            #y2 = points[i - 2][1]
            #cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), thickness=3)  # Draw a line
            # print(points[i-2][0])
            # print(points[i-2][1])

            #cv2.imshow('Frame', frame)

        #print(frame)
        cv2.imshow('Frame', frame)
        # Abort and exit with 'Q'
        key = cv2.waitKey(200)
        if (key == ord('q')):
            break
        elif (key == ord('p')):
            polygon = [np.int32(points)]
            points = []

        cv2.setMouseCallback('Frame', left_click_detect, points)

    capture.release()  # release video file
    cv2.destroyAllWindows()  # close all openCV windows


def redzonecam2(*args):
    print('Camera2 Red Zone saved',args)
    #cv2.displayOverlay('frame', "Overlay 5secs", 5000);
    with open('cam2redzone.txt', 'w') as f:
        f.write(json.dumps(points))

    f.close()
    f = open("cam2redzone.txt", "r")
    list_of_lists=json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol=Polygon(list_of_lists)
    p1=Point(10,20)
    print(pol.contains(p1))
    f.close()
    points.clear()

def yellowzonecam2(*args):
    print('Camera2 Yellow Zone saved',args)
    with open('cam2yellowzone.txt', 'w') as f:
        f.write(json.dumps(points))
    f.close()
    f = open("cam2yellowzone.txt", "r")
    list_of_lists = json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol = Polygon(list_of_lists)
    p1 = Point(10, 20)
    print(pol.contains(p1))
    f.close()
    points.clear()


def greenzonecam2(*args):
    print('Camera2 Green Zone saved',args)
    with open('cam2greenzone.txt', 'w') as f:
        f.write(json.dumps(points))
    f.close()
    f = open("cam2greenzone.txt", "r")
    list_of_lists = json.loads(f.read())
    print(list_of_lists[0])
    list_of_lists.append(list_of_lists[0])
    pol = Polygon(list_of_lists)
    p1 = Point(10, 20)
    print(pol.contains(p1))
    f.close()
    points.clear()


def draw_poly_cam2():
    #CV_GUI_NORMAL = 0x10
    capture = cv2.VideoCapture(1)
    #capture = cv2.VideoCapture("/home/bipun/Downloads/Cam1_2021_07_25-12_00_00_PM.mp4")  # Open video file
    polygon = []
    global points
    points = []
    cv2.namedWindow('Frame',cv2.WINDOW_AUTOSIZE)
    cv2.createButton("RED ZONE", redzonecam2, None, cv2.QT_PUSH_BUTTON, 1)
    cv2.createButton("YELLOW ZONE", yellowzonecam2, None, cv2.QT_PUSH_BUTTON, 1)
    cv2.createButton("GREEN ZONE", greenzonecam2, None, cv2.QT_PUSH_BUTTON, 1)
    while (capture.isOpened()):
        ret, frame = capture.read()  # read a frame
        if not ret:
            print('EOF')
            break
        cv2.polylines(frame, np.array([points]), False, (255,255,0), 3)
        #frame = cv2.polylines(frame, polygon, False, (255, 0, 0), thickness=5)
        i = len(points)
        #if i > 1:
            #x1 = points[i - 1][0]
            #y1 = points[i - 1][1]
            #x2 = points[i - 2][0]
            #y2 = points[i - 2][1]
            #cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), thickness=3)  # Draw a line
            # print(points[i-2][0])
            # print(points[i-2][1])

            #cv2.imshow('Frame', frame)

        #print(frame)
        cv2.imshow('Frame', frame)
        # Abort and exit with 'Q'
        key = cv2.waitKey(200)
        if (key == ord('q')):
            break
        elif (key == ord('p')):
            polygon = [np.int32(points)]
            points = []

        cv2.setMouseCallback('Frame', left_click_detect, points)

    capture.release()  # release video file
    cv2.destroyAllWindows()  # close all openCV windows


but1 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=web1, text='Check Cam1',
              font=('helvetica 15 bold'))
but1.place(x=5, y=104)

but2 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=web2,
              text='Check Cam2', font=('helvetica 15 bold'))
but2.place(x=5, y=176)

but3 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=draw_poly_cam1,
              text='Open Cam1 & Draw', font=('helvetica 15 bold'))
but3.place(x=5, y=250)

but4 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=draw_poly_cam2,
              text='Open Cam2 & Draw', font=('helvetica 15 bold'))
but4.place(x=5, y=322)

#but5 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=blink,
              #text='Detect Eye Blink & Record With Sound', font=('helvetica 15 bold'))
#but5.place(x=5, y=400)

but5 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=GROOVE, text='EXIT', command=exitt,
              font=('helvetica 15 bold'))
but5.place(x=210, y=478)

root.mainloop()
