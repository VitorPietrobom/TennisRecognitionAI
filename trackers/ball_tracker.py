from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    def interpolate_ball_positions(self, ball_detections):
        ball_detections = [x.get(1, []) for x in ball_detections]

        df_ball_detections = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_detections = df_ball_detections.interpolate()
        df_ball_detections = df_ball_detections.fillna(method='bfill')

        ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]

        return ball_detections

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
                cv2.putText(frame, f"Ball", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_frames.append(frame)

        return output_frames
    
    def get_ball_hit_frames(self, ball_detections):
        ball_detections = [x.get(1, []) for x in ball_detections]

        df_ball_detections = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_detections['mid_y'] = (df_ball_detections['y1'] + df_ball_detections['y2']) / 2
        df_ball_detections['mid_y_rolling_mean'] = df_ball_detections['mid_y'].rolling(window=5, min_periods=1, center=False).mean()

        df_ball_detections["delta_y"] = df_ball_detections['mid_y_rolling_mean'].diff()
        df_ball_detections["ball_hit"] = 0
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_detections) - int(minimum_change_frames_for_hit*1.2)):
            negative_position_change = df_ball_detections['delta_y'].iloc[i] > 0 and df_ball_detections['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_detections['delta_y'].iloc[i] < 0 and df_ball_detections['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+ int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_detections['delta_y'].iloc[i] > 0 and df_ball_detections['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_detections['delta_y'].iloc[i] < 0 and df_ball_detections['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count >= minimum_change_frames_for_hit:
                    df_ball_detections['ball_hit'].iloc[i] = 1


        frames_with_ball_hits = df_ball_detections[df_ball_detections["ball_hit"] == 1].index.tolist()
        return frames_with_ball_hits