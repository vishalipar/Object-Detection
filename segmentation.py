# Segmentation - To draw a boundary around the object being tracked

import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-seg.pt')
cap = cv2.VideoCapture('resources\\street.mp4')

while True:
    ret, frame = cap.read()
    results = model.track(frame, classes = [0], persist = True, verbose=False)
    
    for r in results:
        annotated_frame = frame.copy()
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy()
            boxes = r.boxes.xyxy.numpy()
            ids = r.boxes.id.numpy()
            
            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1,y1,x2,y2 = boxes[i].astype(int)
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0,0,255), 2)
                cv2.putText(annotated_frame, f'ID :{person_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                
        annotated_frame = cv2.resize(annotated_frame, (1200, 700))
        cv2.imshow('Object tracking with segmentation', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()