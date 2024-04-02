from ultralytics import YOLO

# model = YOLO('models/yolov5_last.pt')

# results = model.predict('input_videos/input_video.mp4', conf=0.2, save = True)

# print(results)

model = YOLO('yolov8x')

results = model.track('input_videos/input_video.mp4', conf=0.2, save = True)

print(results)