import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

class BackgroundSubstraction():
    def __init__(self, batch_size:int, inner_sample_rate:int=1, method:str = "median"):
        self.batch_size = batch_size
        self.backgrounds = []
        self.method = method 
        self.inner_sample_rate = inner_sample_rate

    
    def _get_batch_background(self, batch_images):
        if self.method == 'median':
            background = np.median(batch_images,axis=0)
        elif self.method == 'mean':
            background = np.mean(batch_images,axis=0)
        return background
        
    
    def fit(self, frame_generator):
        batch_images = []
        for frame_idx, frame in tqdm(enumerate(frame_generator)):
            if (frame_idx % self.inner_sample_rate != 0):
                continue
            if (frame_idx // self.batch_size > len(self.backgrounds)):
                batch_background = self._get_batch_background(batch_images)
                self.backgrounds.append(batch_background)
                batch_images = [] 
            else:
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                batch_images.append(frame_gray)
        if (batch_images):
            batch_background = self._get_batch_background(batch_images)
            self.backgrounds.append(batch_background)
                
                
    def get_background(self, frame_id:int):
        background_id = frame_id // self.batch_size
        return self.backgrounds[background_id]
    
    def visualize(self):
        for idx, background in enumerate(self.backgrounds):
            plt.imshow(background, cmap = 'gray')
            plt.axis("off")
            plt.title(f'Background for images#{self.batch_size*idx} - {self.batch_size*(idx+1)}', 
                      fontsize = 14)
            plt.show()