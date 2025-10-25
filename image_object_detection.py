import cv2
from ultralytics import YOLO

image = cv2.imread('vishal.jpeg')

model = YOLO('yolov8n.pt')

result = model(image)

annotated_frame = result[0].plot()
i = cv2.resize(annotated_frame, (800,800))

cv2.imshow("Object detected", i)

cv2.waitKey(0)
cv2.destroyAllWindows()