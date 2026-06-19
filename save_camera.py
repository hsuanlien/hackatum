import cv2

for i in [0, 1]:
    cap = cv2.VideoCapture(i)

    ret, frame = cap.read()

    print(f"Camera {i}: {ret}")

    if ret:
        cv2.imwrite(f"camera_{i}.jpg", frame)

    cap.release()