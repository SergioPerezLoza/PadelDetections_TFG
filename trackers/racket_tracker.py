from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import pickle  # opcional para guardar detecciones

class RacketTracker:
    def __init__(self, model_path, slice_height=80, slice_width=80, overlap_height_ratio=0.5, overlap_width_ratio=0.5):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.3,
            device='cuda'  # usa 'cpu' si no tienes GPU
        )
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        racket_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                racket_detections = pickle.load(f)
            return racket_detections

        for frame in frames:
            detections = self.detect_frame(frame)
            racket_detections.append(detections)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(racket_detections, f)

        return racket_detections

    def detect_frame(self, frame):
        results = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio
        )

        racket_bboxes = []
        for detection in results.object_prediction_list:
            if detection.category.name.lower() == "raqueta":
                bbox = detection.bbox
                racket_bboxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, detection.score.value))

        return racket_bboxes

    def draw_bboxes(self, frames, detections):
        output_frames = []

        for frame, bboxes in zip(frames, detections):
            for (x1, y1, x2, y2, score) in bboxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Raqueta: {score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames
