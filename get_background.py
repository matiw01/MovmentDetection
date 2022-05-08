import numpy as np
import cv2
import numpy as np
import cv2 as cv

def get_background(cap):
    background = []
    for i in range(300):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        background.append(gray)
        # Display the resulting frame
        # cv.imshow('frame', gray)

    median_frame = np.median(background, axis=0).astype(np.uint8)
    # showing result
    # while True:
    #     cv.imshow('background', median_frame)
    #     if cv.waitKey(1) == ord('q'):
    #         break

    # When everything done, release the capture
    # cap.release()
    cv.destroyAllWindows()

    return median_frame

