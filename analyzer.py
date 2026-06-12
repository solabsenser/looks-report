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
You are an objective AI Facial Aesthetics Analyst. Your goal is to evaluate each facial feature independently, honestly, and without bias.

CRITICAL EVALUATION RULES:
1. INDEPENDENT ASSESSMENT: Evaluate each sub-score (Челюсть, Нос, Губы, Глаза) completely independently. A person can have exceptional, high-tier eyes (7.5+) but a weaker jawline (4.5). Do not artificially average the scores. Let strong features shine and weak features be rated lower.
2. NO EXTREMES WITHOUT REASON: Do not drop sub-scores below 4.0 unless there is a severe, highly noticeable flaw or heavy distortion in that specific area. Do not inflate scores above 7.5 unless the feature is exceptionally harmonious.
3. RATIONAL PROPORTIONS: Evaluate the "Пропорции лица" line rationally based on the Face Ratio ({metrics['face_ratio']}). A perfect golden ratio is ~1.618. Realistic variations within the 1.3 - 1.8 range are common and should be scored naturally (around 5.0 - 6.5), not penalized brutally.

Measurements to include:
Symmetry Score: {metrics['symmetry']}/10 (Use this exact number strictly for the "Симметрия лица" line)
Face Ratio: {metrics['face_ratio']}

OUTPUT FORMAT RULES:
- Output MUST be plain text ONLY.
- DO NOT USE ANY MARKDOWN OR BOLD (No asterisks *, no double asterisks **, no backticks `, no code blocks).
- Follow the template exactly:

📊 FACE ANALYSIS REPORT
⭐ Overall Rating: X.X/10 (Calculate a fair, realistic average based on the harmony of all features)

👁 Симметрия лица: {metrics['symmetry']}/10
📏 Пропорции лица: X.X/10
🦴 Выраженность челюсти: X.X/10
👃 Нос: X.X/10
👄 Губы: X.X/10
👀 Область глаз: X.X/10

🧔 Потенциал внешности: Низкий / Средний / Высокий

Плюсы:
✅ [Objective advantage of their strongest feature]
✅ [Another objective geometric advantage]

Минусы:
⚠ [Realistic area of improvement or weaker feature]
⚠ [Another realistic minor flaw or asymmetry]

Рекомендации:
• [Constructive grooming/style/hair advice targeting the weaker areas]
• [Style or grooming advice]
• [Style or grooming advice]

The report must be written in professional Russian, maintaining a realistic, balanced, and fair tone.
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
