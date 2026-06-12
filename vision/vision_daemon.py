import cv2
import requests
import numpy as np
import json
import time
from ultralytics import YOLO

print("1. Loading YOLOv8 Pose Engine...")
model = YOLO('yolov8n-pose.pt') # Skeletal pose estimation
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
        shelves.append((np.array(shelf["inner"], np.int32), np.array(shelf["outer"], np.int32)))
    print(f"✅ Loaded {len(shelves)} perfectly mapped dual-zone shelves.")
except FileNotFoundError:
    print("❌ Error: zones.json not found. Run zone_mapper.py first!")
    exit()

active_sessions = {}
GRACE_PERIOD = 2.0  
MIN_DWELL_TIME = 1.5  

print("🚀 Vision Daemon Active! (Press 'q' to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="custom_botsort.yaml", classes=0, conf=0.30, verbose=False)
    annotated_frame = results[0].plot()

    for i, (inner, outer) in enumerate(shelves):
        cv2.polylines(annotated_frame, [outer], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(annotated_frame, [inner], True, (0, 0, 255), 2, cv2.LINE_AA)

    current_time = time.time()
    
    # --- THE UPGRADE: Verify both IDs and Skeletons exist ---
    if results[0].boxes.id is not None and results[0].keypoints is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            feet_x, feet_y = int((box[0] + box[2]) / 2), int(box[3])
            cv2.circle(annotated_frame, (feet_x, feet_y), 5, (0, 255, 0), -1)

            # --- SKELETAL GAZE MATH ---
            # Extract Nose (0), Left Shoulder (5), Right Shoulder (6)
            nose_x = kpts[0][0]
            l_shoulder_x = kpts[5][0]
            r_shoulder_x = kpts[6][0]
            
            facing = "CENTER"
            facing_arrow = "|"
            
            # Ensure keypoints are visible to the camera (YOLO assigns 0 if hidden)
            if nose_x > 0 and l_shoulder_x > 0 and r_shoulder_x > 0:
                body_center_x = (l_shoulder_x + r_shoulder_x) / 2
                if nose_x > body_center_x + 15:
                    facing = "RIGHT"
                    facing_arrow = "->"
                elif nose_x < body_center_x - 15:
                    facing = "LEFT"
                    facing_arrow = "<-"

            for i, (inner, outer) in enumerate(shelves):
                in_outer = cv2.pointPolygonTest(outer, (feet_x, feet_y), False) >= 0
                
                if in_outer:
                    # --- THE INTENTION VETO ---
                    # Find the physical center of the shelf zone
                    shelf_center_x = np.mean(inner[:, 0])
                    
                    # If shelf is on their left, but they are looking right -> Ignore
                    if shelf_center_x < feet_x and facing == "RIGHT":
                        continue 
                        
                    # If shelf is on their right, but they are looking left -> Ignore
                    if shelf_center_x > feet_x and facing == "LEFT":
                        continue

                    in_inner = cv2.pointPolygonTest(inner, (feet_x, feet_y), False) >= 0
                    shelf_id = f"Shelf_{i+1}"
                    status = "INTERACTING" if in_inner else "CONSIDERING"

                    if track_id not in active_sessions:
                        active_sessions[track_id] = {
                            "start_time": current_time,
                            "last_seen": current_time, 
                            "shelf": shelf_id,
                            "status": status
                        }
                    else:
                        active_sessions[track_id]["last_seen"] = current_time
                        active_sessions[track_id]["status"] = status
                    
                    dwell = round(current_time - active_sessions[track_id]["start_time"], 1)
                    
                    # Add the Gaze Arrow to the UI so you can verify it is working
                    display_text = f"{status} {facing_arrow} {dwell}s"
                    color = (0, 0, 255) if in_inner else (0, 255, 255)
                    cv2.putText(annotated_frame, display_text, (int(box[0]), int(box[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- THE DEBOUNCER ---
    for t_id in list(active_sessions.keys()):
        session = active_sessions[t_id]
        if current_time - session["last_seen"] > GRACE_PERIOD:
            final_time = round(session["last_seen"] - session["start_time"], 1)
            if final_time >= MIN_DWELL_TIME:
                print(f"📤 SENDING: Shopper {t_id} left {session['shelf']}. Total Dwell: {final_time}s")
                payload = {"shopper_id": t_id, "shelf": session['shelf'], "dwell_time": final_time}
                try: requests.post(BACKEND_URL, json=payload, timeout=1)
                except requests.exceptions.ConnectionError: pass
            del active_sessions[t_id]

    cv2.imshow("ShelfMate AI Vision", annotated_frame)
    if cv2.waitKey(dynamic_delay) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()