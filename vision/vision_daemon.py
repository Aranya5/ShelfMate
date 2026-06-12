import cv2
import requests
import numpy as np
import json
import time # NEW: The stopwatch engine
from ultralytics import YOLO

print("1. Loading YOLOv8 Nano...")
model = YOLO('yolov8n.pt')
BACKEND_URL = "http://localhost:5001/api/shelf-events"

print("2. Loading Test Video Footage...")
cap = cv2.VideoCapture('test.mp4') 
video_fps = cap.get(cv2.CAP_PROP_FPS)
dynamic_delay = int(1000 / video_fps) if video_fps > 0 else 33

try:
    with open('zones.json', 'r') as f:
        zone_data = json.load(f)
    shelves = []
    for shelf in zone_data["shelves"]:
        outer_zone = np.array(shelf["outer"], np.int32)
        inner_zone = np.array(shelf["inner"], np.int32)
        shelves.append((inner_zone, outer_zone))
    print(f"✅ Loaded {len(shelves)} perfectly mapped dual-zone shelves.")
except FileNotFoundError:
    print("❌ Error: zones.json not found. Run zone_mapper.py first!")
    exit()

# --- NEW: The Master Session Database ---
# This remembers who is actively at a shelf and when they got there
active_sessions = {}

print("🚀 Vision Daemon Active! (Press 'q' to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="custom_tracker.yaml", classes=0, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    for i, (inner, outer) in enumerate(shelves):
        cv2.polylines(annotated_frame, [outer], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(annotated_frame, [inner], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(annotated_frame, f"S-{i+1}", tuple(inner[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    current_frame_ids = []
    
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            current_frame_ids.append(track_id)
            feet_x, feet_y = int((x1 + x2) / 2), int(y2)
            cv2.circle(annotated_frame, (feet_x, feet_y), radius=5, color=(0, 255, 0), thickness=-1)

            is_touching_any_zone = False

            for i, (inner, outer) in enumerate(shelves):
                in_outer = cv2.pointPolygonTest(outer, (feet_x, feet_y), False) >= 0
                in_inner = cv2.pointPolygonTest(inner, (feet_x, feet_y), False) >= 0
                
                if in_outer:
                    is_touching_any_zone = True
                    shelf_id = f"Shelf_{i+1}"
                    status = "INTERACTING" if in_inner else "CONSIDERING"

                    # 1. Start Stopwatch if they just entered
                    if track_id not in active_sessions:
                        active_sessions[track_id] = {
                            "start_time": time.time(),
                            "shelf": shelf_id,
                            "status": status
                        }
                    
                    # 2. Update their status live
                    active_sessions[track_id]["status"] = status
                    
                    # 3. Calculate Live Dwell Time
                    dwell_seconds = round(time.time() - active_sessions[track_id]["start_time"], 1)
                    
                    # Draw Live Timer on Screen
                    color = (0, 0, 255) if in_inner else (0, 255, 255)
                    cv2.putText(annotated_frame, f"{status} {dwell_seconds}s", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 4. Exit Event: If they walked away from all shelves, stop the timer and SEND
            if not is_touching_any_zone and track_id in active_sessions:
                session = active_sessions.pop(track_id)
                final_time = round(time.time() - session["start_time"], 1)
                print(f"📤 SENDING: Shopper {track_id} left {session['shelf']}. Total Dwell: {final_time}s")
                
                # Here is where the data goes to Express!
                payload = {"shopper_id": track_id, "shelf": session['shelf'], "dwell_time": final_time}
                try: requests.post(BACKEND_URL, json=payload, timeout=1)
                except requests.exceptions.ConnectionError: pass

    # 5. Cleanup Event: If a shopper completely leaves the camera frame while dwelling
    lost_ids = list(set(active_sessions.keys()) - set(current_frame_ids))
    for lost_id in lost_ids:
        session = active_sessions.pop(lost_id)
        final_time = round(time.time() - session["start_time"], 1)
        print(f"📤 SENDING: Shopper {lost_id} left camera. Total Dwell at {session['shelf']}: {final_time}s")
        
        payload = {"shopper_id": lost_id, "shelf": session['shelf'], "dwell_time": final_time}
        try: requests.post(BACKEND_URL, json=payload, timeout=1)
        except requests.exceptions.ConnectionError: pass

    cv2.imshow("ShelfMate AI Vision", annotated_frame)
    if cv2.waitKey(dynamic_delay) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()