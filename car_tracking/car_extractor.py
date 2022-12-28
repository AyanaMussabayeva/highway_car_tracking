import cv2
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from cached_property import cached_property
from car_tracking.background_subtractor import BackgroundSubstraction


class CarExtractor:
    def __init__(
        self,
        background_sub: BackgroundSubstraction,
        erode_kernel_size: int,
        dilate_kernel_size: int,
        min_contour_area,
    ):
        self.background_sub = background_sub
        self.erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        self.dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        self.min_contour_area = min_contour_area

    def _get_mask(self, mask):
        morph_mask = cv2.erode(mask, self.erode_kernel)
        morph_mask = cv2.dilate(morph_mask, self.dilate_kernel)
        morph_mask = morph_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            image=morph_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )
        image_copy = morph_mask.copy()
        selected_contours = [
            c for c in contours if cv2.contourArea(c) > self.min_contour_area
        ]
        source = np.zeros_like(morph_mask)
        cv2.fillPoly(source, pts=selected_contours, color=(255, 255, 255))
        selected_bboxs = [cv2.boundingRect(c) for c in selected_contours]

        return source, selected_contours, selected_bboxs

    def run(self, frame_gen):
        self.background_sub.fit(frame_gen)
        selected_sources = []
        selected_bboxes = []
        for frame_idx, frame in tqdm(enumerate(frame_gen)):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = self.background_sub.get_background(frame_idx)
            foreground = frame_gray - background
            mask = np.zeros((380, 640))
            mask = np.where(abs(frame_gray - background) > 30, 1, mask)
            selected_source, selected_contours, selected_bboxs = self._get_mask(mask)
            selected_sources.append(selected_source)
            selected_bboxes.append(selected_bboxs)

        return selected_bboxes, selected_sources


class ExtractionReport:
    def __init__(self, frame_gen, selected_bboxes, selected_sources):
        self.frame_gen = frame_gen
        self.selected_bboxes = selected_bboxes
        self.selected_sources = selected_sources

    @cached_property
    def report(self):
        bbox_data = pd.DataFrame()
        for idx, cur_frame_bboxes in enumerate(self.selected_bboxes):
            for bbox_idx, bbox in enumerate(cur_frame_bboxes):
                bbox_data = bbox_data.append(
                    {"bbox_xywh": bbox, "frame": int(idx)}, ignore_index=True
                )
        bbox_data.index.name = "bbox_idx"
        bbox_data["frame"] = bbox_data["frame"].astype(int)
        return bbox_data

    def save_crops(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for idx, frame in tqdm(enumerate(self.frame_gen)):
            if self.selected_bboxes[idx]:
                for bbox_idx, bbox in enumerate(self.selected_bboxes[idx]):
                    x, y, w, h = bbox
                    current_car = frame[y : y + h, x : x + w].copy()
                    filename = f"Bbox_{bbox_idx}_from_frame_{idx}.png"
                    cv2.imwrite(os.path.join(output_dir, filename), current_car)

    def visualize(self):
        pass
