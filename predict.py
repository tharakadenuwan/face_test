"""Face Shape Prediction for Raspberry Pi (TensorFlow Lite).

Usage:
    python predict.py                 # Runs all images in current directory
    python predict.py sample1.png     # Runs a specific image
    python predict.py img1.jpg img2.png # Runs multiple images
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Load TFLite interpreter
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

CLASSES = ["Oblong", "Heart", "Square", "Oval", "Round"]
_LEFT_EYE = (33, 133)
_RIGHT_EYE = (362, 263)
_NOSE_TIP = 1
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454

MODELS_DIR = Path(__file__).resolve().parent / "models"
TFLITE_MODEL_PATH = MODELS_DIR / "faceshape_facenet_v3.tflite"
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"


def load_tflite_and_landmarker():
    """Loads the TFLite interpreter and MediaPipe face landmarker."""
    print("Loading TFLite face shape model...")
    interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()

    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=2,
        )
    )
    return interpreter, landmarker


def _eye_center(landmarks, idxs):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs])
    return pts.mean(axis=0)


def _align(bgr, landmarks):
    h, w = bgr.shape[:2]
    l = _eye_center(landmarks, _LEFT_EYE)
    r = _eye_center(landmarks, _RIGHT_EYE)
    dx, dy = r - l
    angle = np.degrees(np.arctan2(dy, dx))
    mid = ((l + r) / 2) * np.array([w, h])
    M = cv2.getRotationMatrix2D(tuple(mid), angle, 1.0)
    rot = cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rot, tuple(mid.astype(int))


def _is_side_face(landmarks) -> bool:
    nose = landmarks[_NOSE_TIP].x
    d_left = abs(nose - landmarks[_LEFT_CHEEK].x)
    d_right = abs(landmarks[_RIGHT_CHEEK].x - nose)
    if min(d_left, d_right) < 1e-6:
        return True
    return max(d_left, d_right) / min(d_left, d_right) > 2.2


def predict_face_shape(image_path: Path, interpreter, landmarker):
    """Aligns and classifies the face in the image using TFLite."""
    try:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    except Exception as e:
        return {"status": "error", "message": f"Could not read image: {e}"}

    img.thumbnail((1000, 1000))
    rgb = np.asarray(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        return {"status": "no_face", "message": "No face found. Please use a clear, front-facing photo."}
    if len(res.face_landmarks) > 1:
        return {"status": "multiple_faces", "message": "More than one face found. Please use a solo photo."}

    lm = res.face_landmarks[0]
    if _is_side_face(lm):
        return {"status": "side_face", "message": "Face is turned sideways. Please use a straight-on photo."}

    rot, mid = _align(bgr, lm)
    rot_rgb = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
    mp_rot_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rot_rgb))
    res2 = landmarker.detect(mp_rot_img)
    if not res2.face_landmarks:
        return {"status": "no_face", "message": "Could not align face."}

    h, w = rot.shape[:2]
    ew = int(min(w, h) * 0.8)
    eh = int(ew * 1.2)
    x0 = max(0, mid[0] - ew // 2)
    y0 = max(0, mid[1] - int(eh * 0.4))
    crop = rot[y0 : y0 + eh, x0 : x0 + ew]
    if crop.size == 0:
        return {"status": "no_face", "message": "Face crop failed."}

    crop160 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (160, 160)).astype("float32") / 255.0
    crop_tensor = crop160[None, ...]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], crop_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    best_idx = int(np.argmax(probs))
    best_shape = CLASSES[best_idx]
    scores = {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)}

    return {
        "status": "ok",
        "face_shape": best_shape,
        "confidence": scores[best_shape],
        "scores": scores,
    }


def find_all_images(base_dir: Path) -> list[Path]:
    """Finds all supported image files in the directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []
    for file in base_dir.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            # Exclude debug crops
            if not file.name.startswith("debug_"):
                images.append(file)
    return sorted(images)


def process_image(img_path: Path, interpreter, landmarker) -> dict:
    print(f"\nAnalyzing: {img_path.name}...")
    res = predict_face_shape(img_path, interpreter, landmarker)
    print("-" * 45)
    if res["status"] == "ok":
        print(f" Image:       {img_path.name}")
        print(f" Face Shape:  >>> {res['face_shape'].upper()} <<< ({res['confidence']}%)")
        print(" Probabilities:")
        sorted_scores = sorted(res["scores"].items(), key=lambda x: x[1], reverse=True)
        for shape, score in sorted_scores:
            bar = "#" * int(score // 5)
            print(f"   {shape:<7}: {score:>5.1f}%  | {bar}")
    else:
        print(f" Image:       {img_path.name}")
        print(f" Status:      {res['status'].upper()}")
        print(f" Message:     {res['message']}")
    print("-" * 45)
    return res


def main():
    current_dir = Path(__file__).resolve().parent

    # Determine image targets
    if len(sys.argv) > 1:
        targets = []
        for arg in sys.argv[1:]:
            p = Path(arg)
            if not p.is_absolute():
                p = current_dir / p
            if p.exists() and p.is_file():
                targets.append(p)
            else:
                print(f"[WARNING] File not found: {arg}")
    else:
        print("No image specified. Scanning for all images in the folder...")
        targets = find_all_images(current_dir)

    if not targets:
        print("\n[ERROR] No image files found in this folder.")
        print("Please place .jpg or .png images here or specify an image path:")
        print("    python predict.py your_photo.jpg\n")
        sys.exit(1)

    print(f"Found {len(targets)} image(s) to analyze.")
    interpreter, landmarker = load_tflite_and_landmarker()

    results = []
    for img_path in targets:
        result = process_image(img_path, interpreter, landmarker)
        results.append((img_path.name, result))

    # Print summary table if multiple images were processed
    if len(results) > 1:
        print("\n" + "=" * 55)
        print("                  SUMMARY REPORT")
        print("=" * 55)
        print(f" {'IMAGE NAME':<25} | {'FACE SHAPE':<12} | {'CONFIDENCE':<10}")
        print("-" * 55)
        for name, r in results:
            if r["status"] == "ok":
                print(f" {name:<25} | {r['face_shape']:<12} | {r['confidence']}%")
            else:
                print(f" {name:<25} | {r['status'].upper():<12} | -")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
