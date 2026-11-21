from ultralytics import YOLO

# 1. Load a pretrained YOLO model (downloads automatically the first time, ~6 MB)
model = YOLO("yolo26n.pt")

# 2. Run the model on the photo
results = model("fire.jpg")

# 3. Take the result for our single image
result = results[0]

# 4. Go through every box the model found
print(f"Found {len(result.boxes)} objects:")
for box in result.boxes:
    class_id = int(box.cls)            # class number, e.g. 0
    class_name = model.names[class_id] # turn number into name, e.g. "person"
    confidence = float(box.conf)       # how sure the model is (0 to 1)
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # box corners in pixels

    print(f"- {class_name}: {confidence:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# 5. Save a copy of the photo with the boxes drawn on it
result.save("photo_detected.jpg")
print("Saved photo_detected.jpg")