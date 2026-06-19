from ultralytics import YOLO
import supervision as sv
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Tracker
tracker = sv.ByteTrack()

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to get frame")
        break

    # YOLO
    result = model(frame, verbose=False)[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(result)

    # Track
    detections = tracker.update_with_detections(detections)

    # Number of workers currently visible
    current_workers = 0

    if detections.tracker_id is not None:
        current_workers = len(detections.tracker_id)

    # Labels
    labels = []

    if detections.tracker_id is not None:
        for tracker_id in detections.tracker_id:
            labels.append(f"Worker {tracker_id}")

    # Draw boxes
    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Draw labels
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    # Draw worker count
    cv2.putText(
        annotated,
        f"Workers Visible: {current_workers}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("MTU Safety Monitor", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()