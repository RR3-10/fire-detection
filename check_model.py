from ultralytics import YOLO

model = YOLO("best.pt")

# 1. Which training run did this file come from?
args = model.ckpt.get("train_args", {})
print("Run name:", args.get("name"))
print("Epochs:  ", args.get("epochs"))
print("Fraction:", args.get("fraction"))
print("Classes: ", model.names)

# 2. Run with a very low threshold to see what the model "almost" detected
result = model("fire.jpg", conf=0.01, verbose=False)[0]
print(f"\nBoxes with conf >= 0.01: {len(result.boxes)}")
for box in sorted(result.boxes, key=lambda b: -float(b.conf))[:10]:
    print(f"- {model.names[int(box.cls)]}: {float(box.conf):.3f}")