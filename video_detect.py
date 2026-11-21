import cv2
from ultralytics import YOLO

# ---------- Settings ----------
MODEL_PATH = "best.pt"
SOURCE = 0   # a video file, or 0 for your webcam
CONF = 0.35                 # minimum confidence for a box to count
ALERT_FRAMES = 5            # fire must be seen in this many frames in a row
CLEAR_FRAMES = 15

# ---------- Setup ----------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)              # opens the video (or webcam)
if not cap.isOpened():
    raise SystemExit(f"Could not open {SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25       # frames per second of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("fire_video_detected.mp4",       # save the result as a new video
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

streak = 0          # how many frames in a row had a detection
miss = 0
alert_on = False    # is the alarm currently active?
frame_no = 0

# ---------- Main loop: one iteration = one frame ----------
while True:
    ok, frame = cap.read()                  # read the next frame
    if not ok:                              # no more frames = end of video
        break
    frame_no += 1

    result = model(frame, conf=CONF, verbose=False)[0]
    found = len(result.boxes) > 0           # True if fire or smoke was detected

  # --- Alert logic (with hysteresis) ---
    if found:
        streak += 1     # one more frame with fire
        miss = 0        # reset the "no fire" counter
    else:
        streak = 0      # reset the "fire" counter
        miss += 1       # one more frame without fire

    if not alert_on and streak >= ALERT_FRAMES:
        alert_on = True
        print(f"🚨 ALERT at frame {frame_no} ({frame_no / fps:.1f} s)")
        cv2.imwrite(f"alert_{frame_no}.jpg", result.plot())
    elif alert_on and miss >= CLEAR_FRAMES:
        alert_on = False
        print(f"✅ Clear at frame {frame_no} ({frame_no / fps:.1f} s)")

    # --- Draw boxes + status banner ---
    annotated = result.plot()               # frame with boxes drawn
    if alert_on:
        text, color = "FIRE ALERT!", (0, 0, 255)                         # red (BGR)
    elif found:
        text, color = f"Checking... {streak}/{ALERT_FRAMES}", (0, 200, 255)  # orange
    else:
        text, color = "Clear", (0, 200, 0)                               # green
    cv2.rectangle(annotated, (0, 0), (width, 50), color, -1)            # filled bar at the top
    cv2.putText(annotated, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    writer.write(annotated)                                  # add frame to output video
    cv2.imshow("Fire detection - press Q to quit", annotated)  # show it live
    if cv2.waitKey(1) & 0xFF == ord("q"):                    # quit when Q is pressed
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Done: {frame_no} frames processed. Saved fire_video_detected.mp4")