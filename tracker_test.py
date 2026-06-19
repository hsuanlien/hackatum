from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to get frame")
        break

    result = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    detections = tracker.update_with_detections(detections)

    labels = []

    if detections.tracker_id is not None:
        for tracker_id in detections.tracker_id:
            labels.append(f"ID {tracker_id}")

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()