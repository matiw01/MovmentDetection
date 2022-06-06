import cv2 as cv
from get_background import get_background

# path = "http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695"
path = "input/video_3.mp4"
# path = 0
debug = True

import PySimpleGUI as sg

layout = [[sg.Text(" mask coordinates can be hardcoded in detect.py line - 53: (top left corner is 0,0) ")],
          [sg.Text("enter your source:"),
           sg.InputText(default_text=path)],
          [sg.Text("enter masks coordinates: (top left corner is 0,0):"), sg.InputText(default_text="0,0,100,100,100,100,300,300")],
          [sg.Text("sensitivity"),
           sg.InputText(default_text="500")],
          [sg.Button("Debug"), sg.Button("Normal Run")],
          ]

window = sg.Window(title="Motion detection", layout=layout, margins=(100, 50))
while True:
    event, values = window.read()
    path = values[0]
    # End program if user closes window or
    # presses the OK button
    masks = values[1]
    sensitivity = int(values[2])
    if event == "Debug":
        debug = True
        break
    if event == "Normal Run":
        debug = False
        break
window.close()
cap = cv.VideoCapture(path)

masks = masks.split(",")
mask = list(map(int, masks))

if not cap.isOpened():
    print("Cannot open camera ", path)
    exit(1)
else:
    print("opened camera at", path)

background = get_background(cap)
if (debug == True):
    cv.imshow("background", background)
print("got median")

frame_count = 0
consecutive_frame = 8

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv.VideoWriter(
#     "output/result",
#     cv.VideoWriter_fourcc(*'mp4v'), 10,
#     (frame_width, frame_height)
# )


# print(mask[0])

def in_mask(x1, y1, x2, y2, mask):
    n = len(mask)//4
    for i in range(n):
        a, b, c, d = mask[i*4], mask[i*4+1], mask[i*4+2], mask[i*4+3]
        if (a <= x1 <= c and b <= y1 <= d) or (a <= x2 <= c and b <= y2 <= d):
            return True
    return False



while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        orig_frame = frame.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if (debug == True):
            cv.imshow('phase1', gray)
        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        # find the difference between current frame and base frame
        frame_diff = cv.absdiff(gray, background)
        if (debug == True):
            cv.imshow('phase2', frame_diff);
        # thresholding to convert the frame to binary
        ret, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
        if (debug == True):
            cv.imshow('phase3', thres);
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv.dilate(thres, None, iterations=2)
        if (debug == True):
            cv.imshow('phase4', dilate_frame);

        # append the final result into the `frame_diff_list`
        frame_diff_list.append(dilate_frame)
        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)
            # find the contours around the white segmented areas
            contours, hierarchy = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            for i, cnt in enumerate(contours):
                cv.drawContours(frame, contours, i, (0, 0, 255), 3)
            for contour in contours:
                # continue through the loop if contour area is less than 500...
                # ... helps in removing noise detection
                if cv.contourArea(contour) < sensitivity:
                    continue
                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv.boundingRect(contour)
                # draw the bounding boxes
                if not in_mask(x, y, x + w, y + h, mask):
                    cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print(mask)
            # print(mask[0])
            n = len(mask) // 4
            for i in range(n):
                a, b, c, d = mask[i * 4], mask[i * 4 + 1], mask[i * 4 + 2], mask[i * 4 + 3]
                cv.rectangle(orig_frame, (a, b), (c, d), (255, 0, 0), 3)
            # cv.rectangle(orig_farmask)
            cv.imshow('Detected Objects', orig_frame)

            # out.write(orig_frame)
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv.destroyAllWindows()
