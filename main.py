import cv2 as cv

from tracker import Tracker
from tracker import BBox

def main():
    model_path_temp = '../models/model_temp.onnx'
    model_path_frame = '../models/model_frame.onnx'

    tracker = Tracker(model_path_temp, model_path_frame)
    prev_roi = BBox()
    current_roi = BBox()

    video_path = ''
    # video_path = 'C:/Kyle_Work/pysot/demo/bag.avi'

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
        ret, frame = cap.read()

        frame = cv.flip(frame, 1)

        if ret == False:
            continue

        if first_frame:
            init_rect = cv.selectROI(video_name, frame, False, False)
            # init_rect = (310, 141, 119, 118)
            x, y, w, h = init_rect[0], init_rect[1], init_rect[2], init_rect[3]
            prev_roi = BBox(x + w / 2, y + h / 2, w, h)
        
            tracker.preprocessTemp(prev_roi, frame)
            tracker.executeTemp()

            first_frame = False
        else:
            tracker.preprocessFrame(prev_roi, frame)
            tracker.executeFrame()
            bbox, score = tracker.postprocessFrame(prev_roi, frame)

            bbox[2] = (bbox[2] + prev_roi.w) / 2
            bbox[3] = (bbox[3] + prev_roi.h) / 2

            cv.rectangle(frame, (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)), (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)), (0, 255, 0), 3)

            prev_roi = BBox(bbox[0], bbox[1], bbox[2], bbox[3])

            cv.imshow('frame', frame)

            k = cv.waitKey(1)
            if k == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()