import cv2
import numpy as np

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break
    print(f"Frame shape: {frame.shape}")
    # Check average color values (B, G, R)
    b_mean = np.mean(frame[:,:,0])
    g_mean = np.mean(frame[:,:,1])
    r_mean = np.mean(frame[:,:,2])
    print(f"Average BGR values: ({b_mean:.2f}, {g_mean:.2f}, {r_mean:.2f})")
    cv2.imshow('Raw Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
