import cv2
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from cached_property import cached_property
from collections import defaultdict
from car_tracking.car_extractor import ExtractionReport


class Tracker:
    def __init__(self, extraction_report: ExtractionReport, iou_threshold: float):
        self.tracking_df = extraction_report.report
        self.iou_threshold = iou_threshold
        self.tracking_df["track_id"] = self.tracking_df.index

    @cached_property
    def frame2crop(self):
        f2c = defaultdict(list)
        for idx, record in self.tracking_df.iterrows():
            f2c[record["frame"]].append(record["track_id"])
        return f2c

    @staticmethod
    def _calculated_iou(bbox1, bbox2):
        x_tl_1 = bbox1[0]
        x_tl_2 = bbox2[0]
        x_br_1 = x_tl_1 + bbox1[2]
        x_br_2 = x_tl_2 + bbox2[2]
        y_tl_1 = bbox1[1]
        y_tl_2 = bbox2[1]
        y_br_1 = y_tl_1 + bbox1[3]
        y_br_2 = y_tl_2 + bbox2[3]
        x_tl = max(x_tl_1, x_tl_2)
        y_tl = max(y_tl_1, y_tl_2)
        x_br = min(x_br_1, x_br_2)
        y_br = min(y_br_1, y_br_2)

        if x_br < x_tl or y_br < y_tl:
            return 0.0

        intersection_area = (x_br - x_tl) * (y_br - y_tl)
        iou = intersection_area / float(
            bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection_area
        )
        return iou

    def run(self, frame_gen):
        progress_bar = tqdm(range(1, frame_gen.total_frames))
        n_merged = 0
        for frame_id in progress_bar:
            if len(self.frame2crop[frame_id]) == 0:
                continue
            for current_crop in self.frame2crop[frame_id]:
                current_xywh = self.tracking_df.loc[current_crop, "bbox_xywh"]
                for prev_crop in self.frame2crop[frame_id - 1]:
                    prev_xywh = self.tracking_df.loc[prev_crop, "bbox_xywh"]
                    cur_iou = self._calculated_iou(current_xywh, prev_xywh)
                    if cur_iou > self.iou_threshold:
                        self.tracking_df.loc[
                            current_crop, "track_id"
                        ] = self.tracking_df.loc[prev_crop, "track_id"]
                        n_merged += 1
                    progress_bar.set_description(f"Merged {n_merged} tracks")

    def save_by_trackid(self, frame_gen, output_dir, unique_only=False):
        visited_tracks = set()
        n_frames = frame_gen.total_frames // frame_gen.sample_rate
        for frame_idx, current_frame in tqdm(enumerate(frame_gen), total=n_frames):
            cur_frame_df = self.tracking_df[self.tracking_df["frame"] == frame_idx]

            for idx, track_record in cur_frame_df.iterrows():
                track_id = int(track_record["track_id"])
                if unique_only and track_id in visited_tracks:
                    continue
                x, y, w, h = track_record["bbox_xywh"]
                current_car = current_frame[y : y + h, x : x + w]

                trackdir_path = os.path.join(output_dir, f"track_{track_id}")
                os.makedirs(trackdir_path, exist_ok=True)
                filename = f"frame_{frame_idx}.png"
                cv2.imwrite(os.path.join(trackdir_path, filename), current_car)
                visited_tracks.add(track_id)
