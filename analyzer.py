import os
import re
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path(
    os.getenv("FACE_LANDMARKER_MODEL_PATH", "models/face_landmarker.task")
)
MODEL_DOWNLOAD_LOCK = Lock()
ANALYZER_BACKEND = "mediapipe-tasks-face-landmarker-v2"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)


def ensure_face_landmarker_model():
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH

    with MODEL_DOWNLOAD_LOCK:
        if FACE_LANDMARKER_MODEL_PATH.exists():
            return FACE_LANDMARKER_MODEL_PATH

        FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = FACE_LANDMARKER_MODEL_PATH.with_suffix(".tmp")
        urlretrieve(FACE_LANDMARKER_MODEL_URL, temp_path)
        temp_path.replace(FACE_LANDMARKER_MODEL_PATH)

    return FACE_LANDMARKER_MODEL_PATH


def detect_face_landmarks(rgb_image):
    model_path = ensure_face_landmarker_model()

    options = vision.FaceLandmarkerOptions(
        # Force the CPU delegate so MediaPipe does not try to initialize GPU
        # acceleration in headless servers where OpenGL ES libraries are often
        # unavailable. The system package for libGLESv2 is still listed in the
        # deploy files because the MediaPipe wheel can dynamically link to it.
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    return results.face_landmarks[0]


def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = detect_face_landmarks(rgb)

    if landmarks is None:
        return None

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    nose = landmarks[1]

    chin = landmarks[152]
    forehead = landmarks[10]

    face_width = abs(right_eye.x - left_eye.x)
    face_height = abs(chin.y - forehead.y)

    face_ratio = face_height / max(face_width, 0.001)

    eye_center = (left_eye.x + right_eye.x) / 2
    symmetry = 1 - abs(nose.x - eye_center)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    return {
        "face_ratio": round(face_ratio, 2),
        "symmetry": round(symmetry * 10, 1),
        "brightness": round(brightness, 1)
    }


def build_prompt(metrics):
    return f"""
You are a critical aesthetic consultant and facial analysis expert. 
Your task is to provide a realistic, objective, and honest analysis based on the calculated geometric data.

CRITICAL RULES:
1. DO NOT FLATTER. Do not inflate scores out of politeness. Be honest and balanced.
2. Use the standard distribution for scores: an average, normal face with typical minor asymmetries must fall strictly in the 5.0 - 6.0 range. 
3. Scores above 7.0 must be strictly justified by excellent symmetry and near-ideal proportions.
4. The output must be concise and straightforward. Do not include introductory or concluding remarks.

Measurements to evaluate:
Symmetry Score: {metrics['symmetry']}/10 (Use this directly as a baseline for the symmetry line)
Face Ratio: {metrics['face_ratio']} (Ideal is ~1.618 golden ratio. Deviations should lower the proportion score)
Brightness: {metrics['brightness']}

Format the output strictly as plain text matching the lines below. 
DO NOT USE ANY MARKDOWN (no asterisks, no bold text, no code blocks like ` or **).

📊 FACE ANALYSIS REPORT
⭐ Overall Rating: X.X/10

👁 Симметрия лица: {metrics['symmetry']}/10
📏 Пропорции лица: X.X/10
🦴 Выраженность челюсти: X.X/10
👃 Нос: X.X/10
👄 Губы: X.X/10
👀 Область глаз: X.X/10

🧔 Потенциал внешности: Низкий / Средний / Высокий

Плюсы:
✅ [Specific objective advantage based on metrics or photo]
✅ [Specific objective advantage]

Минусы:
⚠ [Real critique or area of improvement]
⚠ [Real critique or area of improvement]

Рекомендации:
• [Constructive style/grooming advice to improve appearance]
• [Style/grooming advice]
• [Style/grooming advice]

Ensure the report remains professional and constructive in Russian, but strictly uncompromising and critical.
"""


def extract_score(report):
    match = re.search(
        r"Overall Rating:\s*(\d+(?:\.\d+)?)",
        report
    )

    if match:
        try:
            return float(match.group(1))
        except Exception:
            return 0.0

    return 0.0


def analyze_face(image_path):
    try:
        metrics = calculate_face_metrics(image_path)

        if metrics is None:
            return {
                "score": 0,
                "report": (
                    "❌ Не удалось обнаружить лицо.\n\n"
                    "Попробуйте загрузить фото:\n"
                    "• анфас\n"
                    "• хорошее освещение\n"
                    "• один человек в кадре\n"
                    "• лицо должно быть полностью видно"
                )
            }

        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(
            build_prompt(metrics)
        )

        report = response.text.strip()

        score = extract_score(report)

        return {
            "score": score,
            "report": report
        }

    except Exception as e:
        return {
            "score": 0,
            "report": (
                "❌ Ошибка анализа изображения.\n\n"
                f"Детали: {str(e)}"
            )
        }


if __name__ == "__main__":
    result = analyze_face("photo.jpg")

    print("SCORE:")
    print(result["score"])

    print("\nREPORT:")
    print(result["report"])
