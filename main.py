import cv2 as cv
import numpy as np
import math

from tracker import Tracker
from tracker import BBox

def main():
    # model paths
    model_path_temp = '../models/model_temp.onnx'
    model_path_frame = '../models/model_frame.onnx'

    # initalizing variables
    tracker = Tracker(model_path_temp, model_path_frame)
    prev_roi = BBox()
    trajectory = []

    # setting up the video
    video_path = input('Please give a video path (blank if using webcam): ')
    if video_path == '':
        video_name = "Webcam"
        cap = cv.VideoCapture(0)
    else:
        video_name = video_path.split('/')[-1].split('.')[0]
        cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Unable to get frame')

    first_frame = True
    while True:
        # getting the frame
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if ret == False:
            continue

        if first_frame:
            # selecting the area of interest
            init_rect = cv.selectROI(video_name, frame, False, False)
            if (init_rect == (0, 0, 0, 0)): break
            x, y, w, h = init_rect[0], init_rect[1], init_rect[2], init_rect[3]
            prev_roi = BBox(x + w / 2, y + h / 2, w, h)
            trajectory.append([int(prev_roi.xc), int(prev_roi.yc)])
        
            # getting tensor outputs from data
            tracker.preprocessTemp(prev_roi, frame)
            tracker.executeTemp()

            first_frame = False
        else:
            # getting tensor outputs from data
            tracker.preprocessFrame(prev_roi, frame)
            tracker.executeFrame()
            bbox, score = tracker.postprocessFrame(prev_roi, frame)

            # updating trajectory
            prev_roi = BBox(bbox[0], bbox[1], (bbox[2] + prev_roi.w) / 2, (bbox[3] + prev_roi.h) / 2)
            trajectory.append([int(prev_roi.xc), int(prev_roi.yc)])

            # drawing roi, trajectory, and starting point
            cv.rectangle(frame, (int(prev_roi.xc - prev_roi.w / 2), int(prev_roi.yc - prev_roi.h / 2)), (int(prev_roi.xc +  prev_roi.w / 2), int( prev_roi.yc +  prev_roi.h / 2)), (0, 255, 0), 3)
            cv.polylines(frame, [np.array(trajectory, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)
            cv.circle(frame, trajectory[0], 3, (0, 0, 255), -1)
            cv.imshow('frame', frame)

            # resetting trajectory if the object comes back to starting point
            if (len(trajectory) > 10):
                if (math.dist(trajectory[-1], trajectory[0]) < math.dist(trajectory[-10], trajectory[0]) \
                and abs(trajectory[-1][0] - trajectory[0][0]) <= 10 and abs(trajectory[-1][1] - trajectory[0][1]) <= 10):
                    trajectory = [trajectory[-1]]

            k = cv.waitKey(1)
            if k == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()