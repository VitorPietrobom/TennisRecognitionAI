from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    def detect_frame(self, frame):
        results = self.model.predict(frame,conf=0.15)[0]


        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections


        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    def draw_boxes(self, frames, ball_detections):
        output_frames = []
        for frame, player_dict in zip(frames, ball_detections):
            for track_id, box in player_dict.items():
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Ball {track_id}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_frames.append(frame)

        return output_frames