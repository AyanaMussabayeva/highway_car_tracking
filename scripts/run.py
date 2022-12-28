import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm.auto import tqdm
import pandas as pd

from car_tracking.frame_generator import FrameGen
from car_tracking.background_subtractor import BackgroundSubstraction
from car_tracking.car_extractor import CarExtractor
from car_tracking.car_extractor import ExtractionReport
from car_tracking.tracker import Tracker

if __name__ == "__main__":

	video_path = 'data/highway.mp4'
	sample_rate = 2 

	gen = FrameGen(video_path, sample_rate, 50)

	bs = BackgroundSubstraction(batch_size=25, inner_sample_rate = 2)
	bs.fit(gen)

	car_extract = CarExtractor(bs, 3, 9, 500)
	selected_bboxes, selected_sources  = car_extract.run(gen)

	extraction_report = ExtractionReport(gen, selected_bboxes, selected_sources)
	extraction_report.report
	extraction_report.save_crops('outputs/saved_crops/')

	tracker = Tracker(extraction_report, 0.05)
	tracker.run(gen)
	tracker.save_by_trackid(gen, 'outputs/tracks/')
