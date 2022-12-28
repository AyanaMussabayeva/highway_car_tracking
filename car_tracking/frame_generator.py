import cv2


class FrameGen():
    def __init__(self, video_path, sample_rate, total_frames=None):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.total_frames = total_frames
        self.cap = cv2.VideoCapture(self.video_path)
        if (self.total_frames == None):
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
    def __iter__(self):
        for fno in range(0, self.total_frames, self.sample_rate):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 380))
            yield frame

    