import cv2
import requests
from ultralytics import YOLO

print("1. Loading YOLOv8 Nano...")
# This will download the tiny .pt weights file locally on the first run
model = YOLO('yolov8n.pt')

# This is where we will point it to your Express server later
BACKEND_URL = "http://localhost:5001/api/shelf-events"

print("2. Loading Test Video Footage...")
# Feed the static video file into the OpenCV capture engine
cap = cv2.VideoCapture('store_aisle.mp4')

print("🚀 Vision Daemon Active! (Press 'q' in the video window to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab camera frame.")
        break
    # # Flip the array horizontally
    # frame = cv2.flip(frame, 1)

    # Run YOLO on the live frame
    results = model(frame, verbose=False)
    people_count = 0

    # Let YOLO draw its own bounding boxes on a copy of the frame
    annotated_frame = results[0].plot()

    # Extract the raw JSON data for the backend
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == 'person' and confidence > 0.65:
            people_count += 1

    # Show the live video feed on your screen
    cv2.imshow("ShelfMate AI Vision", annotated_frame)

# Console logging logic 
    if people_count > 0:
        print(f"📡 Event Triggered: {people_count} shopper(s) detected.")
        
        # The Armor: Try to send the data, but don't crash if the server is down
        try:
            requests.post(BACKEND_URL, json={"count": people_count}, timeout=1)
        except requests.exceptions.ConnectionError:
            print("⚠️ Backend offline. Telemetry dropped, but vision remains active.")

    # Listen for the 'q' key to shut down gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera released safely.")