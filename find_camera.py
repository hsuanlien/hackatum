import cv2

for i in range(10):
    print(f"\nTesting camera {i}")

    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print("Not opened")
        continue

    ret, frame = cap.read()

    print("Frame captured:", ret)

    cap.release()