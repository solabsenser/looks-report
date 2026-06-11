import os
import re
import cv2

# Самый надежный и прямой импорт решений face_mesh напрямую из пакета
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)


def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Используем напрямую импортированный модуль face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

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
You are a facial analysis assistant.

Analyze only visible facial characteristics and image quality.

Measurements:

Symmetry Score: {metrics['symmetry']}
Face Ratio: {metrics['face_ratio']}
Brightness: {metrics['brightness']}

IMPORTANT:

Return the report in Russian.

The first line MUST be:

⭐ Overall Rating: X.X/10

Use this exact wording.

Format exactly:

📊 FACE ANALYSIS REPORT

⭐ Overall Rating: X.X/10

👁 Симметрия лица: X.X/10
📏 Пропорции лица: X.X/10
🦴 Выраженность челюсти: X.X/10
👃 Нос: X.X/10
👄 Губы: X.X/10
👀 Область глаз: X.X/10

🧔 Потенциал внешности: Низкий / Средний / Высокий

Плюсы:
✅ пункт
✅ пункт

Минусы:
⚠ пункт
⚠ пункт

Рекомендации:
• пункт
• пункт
• пункт

Keep the report constructive.
Do not insult the user.
Do not use markdown.
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
