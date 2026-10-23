# Face Shape Prediction (Raspberry Pi Edition)

Lightweight, optimized face shape classification powered by TensorFlow Lite and MediaPipe.

---

## Folder Contents (Everything needed for Raspberry Pi)

- **`models/`**
  - `face_landmarker.task` — MediaPipe 468-point face landmarker.
  - `faceshape_facenet_v3.tflite` — Optimized 23.4 MB FaceNet model (replaces the 135 MB Keras model).
- **`predict.py`** — Fast TFLite prediction script. Automatically analyzes all images in the folder, or single images.
- **`run.sh`** — One-click runner for Raspberry Pi / Linux.
- **`run.bat`** — One-click runner for Windows testing.
- **`requirements.txt`** — Clean list of Python dependencies.
- **`sample1.png`** — Test face image.

---

## How to Run on Raspberry Pi

### 1. Requirements
- **Board:** Raspberry Pi 4 or Raspberry Pi 5.
- **OS:** Raspberry Pi OS **64-bit** *(Debian Bookworm or Bullseye)*.

### 2. Initial Setup (One-time)
Open terminal on your Raspberry Pi:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0
```

### 3. Run Predictions

Make `run.sh` executable and run it:
```bash
chmod +x run.sh
./run.sh
```

- **Run all images in folder:**
  ```bash
  ./run.sh
  ```
- **Run a specific image:**
  ```bash
  ./run.sh my_photo.jpg
  ```
- **Or directly with Python:**
  ```bash
  python3 predict.py
  python3 predict.py my_photo.jpg
  ```
