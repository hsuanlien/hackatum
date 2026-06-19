from ultralytics import YOLO


class HelmetDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_helmets(self, frame):
        """
        Returns helmet boxes as [(x1, y1, x2, y2, confidence), ...]
        NOTE: yolov8n.pt does NOT detect helmets.
        Replace model_path with a trained helmet model later.
        """
        results = self.model(frame, verbose=False)[0]

        helmet_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            # Change these names depending on your helmet model classes
            if class_name.lower() in ["helmet", "hardhat", "hard_hat", "safety helmet"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                helmet_boxes.append((x1, y1, x2, y2, conf))

        return helmet_boxes


def helmet_inside_person(person_bbox, helmet_bbox):
    px1, py1, px2, py2 = person_bbox
    hx1, hy1, hx2, hy2, _ = helmet_bbox

    helmet_center_x = (hx1 + hx2) // 2
    helmet_center_y = (hy1 + hy2) // 2

    # Helmet should be inside person box
    inside_person = (
        px1 <= helmet_center_x <= px2 and
        py1 <= helmet_center_y <= py2
    )

    # Helmet should be near top 40% of body
    person_height = py2 - py1
    top_region_bottom = py1 + int(0.4 * person_height)

    in_head_region = py1 <= helmet_center_y <= top_region_bottom

    return inside_person and in_head_region