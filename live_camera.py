"""Live face-shape classification using the existing TFLite and MediaPipe models.
Press q or ESC to quit the camera window.
"""
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

CLASSES = ["Oblong", "Heart", "Square", "Oval", "Round"]
LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "faceshape_facenet_v3.tflite"
LANDMARKER_PATH = BASE_DIR / "models" / "face_landmarker.task"


def load_models():
    if not MODEL_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Keep the models folder beside this script.")
    print("Loading face-shape model...")
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )
    return interpreter, landmarker


def eye_center(landmarks, indexes):
    return np.mean([[landmarks[i].x, landmarks[i].y] for i in indexes], axis=0)


def predict_frame(frame, interpreter, landmarker):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    landmarks = result.face_landmarks[0]

    nose = landmarks[NOSE_TIP].x
    d_left = abs(nose - landmarks[LEFT_CHEEK].x)
    d_right = abs(landmarks[RIGHT_CHEEK].x - nose)
    if min(d_left, d_right) < 1e-6 or max(d_left, d_right) / min(d_left, d_right) > 2.2:
        return "Face camera directly", None

    left = eye_center(landmarks, LEFT_EYE)
    right = eye_center(landmarks, RIGHT_EYE)
    dx, dy = right - left
    angle = np.degrees(np.arctan2(dy, dx))
    midpoint = ((left + right) / 2) * np.array([w, h])
    matrix = cv2.getRotationMatrix2D(tuple(midpoint), angle, 1.0)
    aligned = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    ew = int(min(w, h) * 0.8)
    eh = int(ew * 1.2)
    cx, cy = midpoint.astype(int)
    x0 = max(0, cx - ew // 2)
    y0 = max(0, cy - int(eh * 0.4))
    crop = aligned[y0:min(h, y0 + eh), x0:min(w, x0 + ew)]
    if crop.size == 0:
        return "Could not crop face", None

    crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (160, 160))
    tensor = crop.astype(np.float32) / 255.0
    tensor = tensor[None, ...]
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_info["index"], tensor)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_info["index"])[0]
    best = int(np.argmax(scores))
    return CLASSES[best], float(scores[best]) * 100


def main():
    interpreter, landmarker = load_models()
    # Use Picamera2 for the Raspberry Pi CSI camera module.
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    camera.start()

    print("Camera started. Press q or ESC in the camera window to quit.")
    label = "Looking for face..."
    confidence = None
    frame_count = 0
    try:
        while True:
            rgb_frame = camera.capture_array()
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            frame_count += 1
            # Run the neural model every 3 frames to reduce CPU load.
            if frame_count % 3 == 0:
                try:
                    label, confidence = predict_frame(frame, interpreter, landmarker)
                    if label is None:
                        label, confidence = "No face detected", None
                except Exception as exc:
                    label, confidence = "Detection error", None
                    print("Detection error:", exc)

            text = label if confidence is None else f"{label} ({confidence:.1f}%)"
            cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 440), 55), (25, 25, 25), -1)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Live Face Shape Detection - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
