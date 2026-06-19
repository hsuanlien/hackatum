from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # change to 1 later if needed

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to get frame")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()