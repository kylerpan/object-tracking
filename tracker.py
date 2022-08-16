import cv2 as cv
import numpy as np
import onnxruntime as rt

class BBox:
    def __init__(self, xc = 0, yc = 0, w = 0, h = 0):
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

class Tracker:
    def __init__(self, model_path_temp, model_path_frame):
        self.temp_session = rt.InferenceSession(model_path_temp, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.temp_inputs = [x.name for x in self.temp_session.get_inputs()]
        self.temp_outputs = [x.name for x in self.temp_session.get_outputs()]
        self.temp_output_tensors = []

        self.frame_session = rt.InferenceSession(model_path_frame, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.frame_inputs = [x.name for x in self.frame_session.get_inputs()]
        self.frame_outputs = [x.name for x in self.frame_session.get_outputs()]
        self.frame_output_tensors = []

        self.scale_ratio_ = 8
        self.num_anchors_ = 5
        self.anchors = np.load('anchors.npy')
        self.score_size_ = 25
        self.scale = 0

    def getSubWindow(self, frame, center, model_sz, original_sz, avg_chans):
        r, c, k = frame.shape
        temp = original_sz / 2

        context_xmin = int(center[0] - temp)
        context_xmax = context_xmin + original_sz
        context_ymin = int(center[1] - temp)
        context_ymax = context_ymin + original_sz

        left_pad = 0
        top_pad = 0

        if (context_xmin < 0): 
            left_pad = -context_xmin
            context_xmin = 0
        if (context_ymin < 0): 
            top_pad = -context_ymin
            context_ymin = 0
        if (context_xmax > c - 1): context_xmax = c - 1
        if (context_ymax > r - 1): context_ymax = r - 1

        im_patch = np.full((original_sz, original_sz, k), avg_chans, dtype=np.ubyte)
        im_patch[top_pad:top_pad + context_ymax - context_ymin, left_pad:left_pad + context_xmax - context_xmin, :] = frame[context_ymin:context_ymax, context_xmin:context_xmax, :]

        # cv.imshow('patch', im_patch)
        # cv.waitKey(1)
        
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)

        return im_patch

    def preprocessTemp(self, roi, frame):
        w_z = roi.w + (roi.w + roi.h) / 2
        h_z = roi.h + (roi.w + roi.h) / 2
        s_z = round(np.sqrt(w_z * h_z))

        self.z_crop = self.getSubWindow(frame, (roi.xc, roi.yc), 127, s_z, 127)

    def executeTemp(self):
        onnx_inputs = {self.temp_inputs[0]: self.z_crop}
        self.temp_output_tensors = self.temp_session.run(self.temp_outputs, onnx_inputs)

    def preprocessFrame(self, roi, frame):
        w_z = roi.w + (roi.w + roi.h) / 2
        h_z = roi.h + (roi.w + roi.h) / 2
        s_z = 2 * round(np.sqrt(w_z * h_z))
        self.scale = 255 / s_z

        self.z_crop = self.getSubWindow(frame, (roi.xc, roi.yc), 255, s_z, 127)

    def executeFrame(self):
        onnx_inputs = {self.frame_inputs[0]: self.z_crop, self.frame_inputs[1]: self.temp_output_tensors[0], self.frame_inputs[2]: self.temp_output_tensors[1], self.frame_inputs[3]: self.temp_output_tensors[2]}
        self.frame_output_tensors = self.frame_session.run(self.frame_outputs, onnx_inputs)

    def postprocessFrame(self, prev_roi, frame):
        score = np.reshape(np.transpose(np.reshape(self.frame_output_tensors[0], (2, 5, 25, 25)), (1, 2, 3, 0)), (3125, 2))
        score -= np.amax(score)
        self.score = np.exp(score[:, 1]) / np.sum(score[:, 0]) + np.exp(score[:, 1])

        self.pred_bbox = np.reshape(np.transpose(np.reshape(self.frame_output_tensors[1], (4, 5, 25, 25)), (1, 2, 3, 0)), (3125, 4))
        self.pred_bbox[:, 0] = self.pred_bbox[:, 0] * self.anchors[:, 2] + self.anchors[:, 0]
        self.pred_bbox[:, 1] = self.pred_bbox[:, 1] * self.anchors[:, 3] + self.anchors[:, 1]
        self.pred_bbox[:, 2] = np.exp(self.pred_bbox[:, 2]) * self.anchors[:, 2]
        self.pred_bbox[:, 3] = np.exp(self.pred_bbox[:, 3]) * self.anchors[:, 3]

        # aspect ratio
        ratio0 = prev_roi.w / prev_roi.h
        ratio1 = self.pred_bbox[:, 2] / self.pred_bbox[:, 3]
        r_c = np.maximum(ratio0/ratio1, ratio1/ratio0)

        # scale ratio
        scale0 = (prev_roi.w + prev_roi.h) / 2
        scale1 = (self.pred_bbox[:, 2] + self.pred_bbox[:, 3]) / 2
        s_c = np.maximum(scale0/scale1, scale1/scale0)

        wts = np.exp(-(r_c * s_c - 1) * 0.05)
        self.score = self.score * wts

        best_idx = np.argmax(self.score)
        bbox = self.pred_bbox[best_idx, :]
        bbox /= self.scale

        x0 = bbox[0] + prev_roi.xc - prev_roi.w / 2
        y0 = bbox[1] + prev_roi.yc - prev_roi.h / 2
        x1 = x0 + bbox[2]
        y1 = y0 + bbox[3]

        r, c, k = frame.shape
        x0 = max(0, min(x0, c))
        y0 = max(0, min(y0, r))
        x1 = max(0, min(x1, c))
        y1 = max(0, min(y1, r))

        return [[(x0 + x1) / 2 , (y0 + y1) / 2, x1 - x0, y1 - y0], self.score[best_idx]]

