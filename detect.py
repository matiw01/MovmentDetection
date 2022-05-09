import cv2 as cv
from get_background import get_background

path = "http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695"
debug =True
cap = cv.VideoCapture(path)
print("captured")

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

background = get_background(cap)
if (debug==True):
    cv.imshow("background",background);
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
mask = (0, 0, 5000, 5000)


def in_mask(x1, y1, x2, y2, mask):
    a, b, c, d = mask
    return (a <= x1 <= c and b <= y1 <= d) or (a <= x2 <= c and b <= y2 <= d)


while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        orig_frame = frame.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if(debug==True):
            cv.imshow('phase1', gray)
        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        # find the difference between current frame and base frame
        frame_diff = cv.absdiff(gray, background)
        if(debug==True):
            cv.imshow('phase2', frame_diff);
        # thresholding to convert the frame to binary
        ret, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
        if(debug==True):
            cv.imshow('phase3', thres);
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv.dilate(thres, None, iterations=3)
        if(debug==True):
            cv.imshow('phase4', thres);

        # append the final result into the `frame_diff_list`
        frame_diff_list.append(thres)
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
                if cv.contourArea(contour) < 500:
                    continue
                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv.boundingRect(contour)
                # draw the bounding boxes
                if not in_mask(x, y, x+w, y+h, mask):
                    cv.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv.imshow('Detected Objects', orig_frame)

            # out.write(orig_frame)
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv.destroyAllWindows()