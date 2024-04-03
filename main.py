from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    input_video_path = 'input_videos/input_video.mp4'
    output_video_path = 'output_videos/output_video.avi'
    frames = read_video(input_video_path)

    player_tracker = PlayerTracker(model_name="yolov8x")
    player_detections = player_tracker.detect_frames(frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    ball_tracker = BallTracker(model_name="models/yolov5_last.pt")
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    court_line_detector = CourtLineDetector(model_path="models/keypoints_model.pth")
    court_line_detections = court_line_detector.predict(frames[0])


    frames = ball_tracker.draw_boxes(frames, ball_detections)
    frames = player_tracker.draw_boxes(frames, player_detections)
    frames = court_line_detector.draw_keypoints_on_video(frames, court_line_detections)

    ## Drawing frame number in the top left corner
    for i, frame in enumerate(frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(frames, output_video_path, 24)
    

if __name__ == "__main__":
    main()