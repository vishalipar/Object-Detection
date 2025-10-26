# Trail :- To draw a connecting line in the center of the detected object rectangle

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('resources\\people_walking.mp4') # \\ to avoid \p as excape sequence

id_map = {}
next_id = 1

trail = defaultdict(lambda :deque(maxlen=30)) # maxlen defines the length of the trail line
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    res = model.track(frame, classes = [0], persist = True, verbose=False)
    annotated_frame = frame.copy()
    
    if res[0].boxes and res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()
        
        for box, oid in zip(boxes, ids):
            x1,y1,x2,y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            appear[oid] += 1
            
            if appear[oid] >=5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1
                
            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx,cy))
                
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (255,0,0), 1)
                
                cv2.putText(annotated_frame, f'ID :{sid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2)
                
                cv2.circle(annotated_frame, (cx, cy), 5, (0,255,0), -1)
                
                trail_points = list(trail[oid])
                for i in range(1, len(trail_points)):
                    # connect the trail points to make a trail line
                    cv2.line(annotated_frame, trail_points[i-1], trail_points[i], (0,255,0), 2)
                    
    annotated_frame = cv2.resize(annotated_frame, (1200,700)) # resize the image size
    cv2.imshow('Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()