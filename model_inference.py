# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
import cv2
from  ultralytics import YOLO
from PIL import Image
import numpy as np

from config import odo_region_detect_model_path
from odometer_ocr_model import OMOcrModel


class DetectOMRegionAndOCR:
    def __init__(self, model_path=None, ocr_model_path=None):
        if not model_path:
            model_path = odo_region_detect_model_path
        self.odo_region_detect_model = YOLO(model_path,
                                            verbose=False)
        self.odo_orm_model = OMOcrModel(ocr_model_path)

    def predict(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = self.odo_region_detect_model(img)

        # Extract bounding boxes from the results
        # boxes = results.pandas().xyxy[0]
        classes = results[0].boxes.cls  # Class labels
        confidences = results[0].boxes.conf
        boxes = results[0].boxes.xyxy
        # Assuming one image for simplicity

        # Iterate through each detected object
        for cls, box, conf in zip(classes, boxes, confidences):

            # Check for minimum confidence threshold (optional)
            if conf > 0.5:  # Adjust this threshold as needed
                # Crop the image based on bounding box coordinates
                crop_obj = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                name = self.odo_region_detect_model.names[int(cls)]
                # cv2.imshow(f"cropped_object_{name}", crop_obj)
                if name == "odometer":
                    return self.odo_orm_model.predict(crop_obj)
        return ""
