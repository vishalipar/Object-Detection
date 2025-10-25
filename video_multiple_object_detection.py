import cv2
from ultralytics import YOLO

video = cv2.VideoCapture('busy_street.mp4')

model = YOLO('yolov8n.pt')

while True:
    ret, frame = video.read()
    result = model(frame)
    annotated_video = result[0].plot()
    
    cv2.imshow('Video object detection ', annotated_video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()