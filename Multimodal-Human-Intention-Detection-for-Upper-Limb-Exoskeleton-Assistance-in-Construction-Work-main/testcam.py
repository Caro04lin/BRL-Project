#Vizualize camera stream (Ordro EP6 PLUS)

import cv2

# Replace with your camera's RTSP/HTTP URL
camera_url = "rtsp://192.168.1.1:554/live.sdp"

# Open the video stream
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Camera connected successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the video frame
    cv2.imshow('Camera Stream', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()