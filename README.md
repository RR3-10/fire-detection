# 🔥 Fire & Smoke Detection with YOLO26

A real-time fire and smoke detector. I fine-tuned **YOLO26n** on the **D-Fire** dataset (21,527 images), and the app detects fire and smoke in images, videos and live webcam footage. It raises an alarm only when fire persists across several frames.

## Results (test set, 4,302 images never used in training)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| smoke | 0.837 | 0.772 | 0.834 | 0.516 |
| fire  | 0.730 | 0.648 | 0.717 | 0.370 |
| **all** | **0.783** | **0.710** | **0.775** | **0.443** |

- Validation mAP50 (0.784) is close to test mAP50 (0.775), so the model generalizes well.
- The pretrained YOLO26n detected **no fire** in a burning-house photo. After fine-tuning, it detects the fire at 0.80 confidence.

## How it works
1. **Data:** D-Fire has 17,221 training images, which I split into 14,638 for training and 2,583 for validation (15%, seed 42). It also has 4,306 test images. About 45% of the images contain no fire, which teaches the model to ignore sunsets and lights.
2. **Training:** YOLO26n for 40 epochs at 640 px on a Colab T4 GPU (2.7 hours). Hue augmentation is kept very low (`hsv_h=0.005`), because colour is a key feature of fire.
3. **Alarm logic with hysteresis:** the alarm turns on after fire appears in **5 consecutive frames**, and turns off only after **15 consecutive frames without fire**. This stops the alarm from flickering on and off when the model misses a single frame.

## Project structure
| File | Purpose |
|---|---|
| `notebooks/fire_detection.ipynb` | Data exploration, training and evaluation (Colab) |
| `best.pt` | The trained model |
| `app.py` | Streamlit web app (image and video upload) |
| `video_detect.py` | Video or webcam detection with alarm and snapshots |
| `first_detection.py` | First experiment with the pretrained YOLO |
| `test_my_model.py`, `check_model.py` | Testing and debugging the trained model |

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py