import cv2
import requests
from ultralytics import YOLO

print("1. Loading YOLOv8 Nano...")
model = YOLO('yolov8n.pt')

BACKEND_URL = "http://localhost:5001/api/shelf-events"

print("2. Loading Test Video Footage...")
cap = cv2.VideoCapture('store_aisle.mp4')

video_fps = cap.get(cv2.CAP_PROP_FPS)
dynamic_delay = int(1000 / video_fps) if video_fps > 0 else 33

print("🚀 Vision Daemon Active! (Press 'q' in the video window to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab camera frame.")
        break

    # --- THE UPGRADE: Stateful Tracking ---
    # persist=True gives memory. classes=0 filters ONLY humans. conf=0.65 applies your threshold.
    results = model.track(frame, persist=True, classes=0, conf=0.65, verbose=False)
    
    # Let YOLO draw its bounding boxes AND tracking IDs
    annotated_frame = results[0].plot()

    # Extract the unique tracking IDs for this specific frame
    active_shopper_ids = []
    if results[0].boxes.id is not None:
        # Convert the tensor of IDs to a standard Python list
        active_shopper_ids = results[0].boxes.id.int().cpu().tolist()

    people_count = len(active_shopper_ids)

    # Show the live video feed on your screen
    cv2.imshow("ShelfMate AI Vision", annotated_frame)

    # --- Telemetry & Backend Logging ---
    if people_count > 0:
        # Now we can see EXACTLY who is in the frame
        print(f"📡 Event: {people_count} shopper(s) detected. IDs: {active_shopper_ids}")
        
        try:
            # We will eventually send the IDs here for dwell time math
            requests.post(BACKEND_URL, json={"count": people_count}, timeout=1)
        except requests.exceptions.ConnectionError:
            print("⚠️ Backend offline. Telemetry dropped, but vision remains active.")

    if cv2.waitKey(dynamic_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera released safely.")