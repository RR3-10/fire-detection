from ultralytics import YOLO

model = YOLO("best.pt")          # YOUR trained model, not the pretrained one

results = model("fire.jpg")      # the burning-house photo from Stage 1
result = results[0]

print(f"Found {len(result.boxes)} objects:")
for box in result.boxes:
    class_name = model.names[int(box.cls)]   # now this is "smoke" or "fire"
    confidence = float(box.conf)
    print(f"- {class_name}: {confidence:.2f}")

result.save("fire_detected.jpg")
print("Saved fire_detected.jpg")