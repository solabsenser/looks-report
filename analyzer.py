import cv2
import mediapipe as mp
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

mp_face_mesh = mp.solutions.face_mesh


def calculate_face_metrics(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    h, w = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

Based ONLY on these visible image measurements:

Symmetry Score: {metrics['symmetry']}
Face Ratio: {metrics['face_ratio']}
Brightness: {metrics['brightness']}

Create a report in EXACTLY this format:

📊 FACE ANALYSIS REPORT

⭐ Overall Rating: X.X/10

👁 Facial Symmetry: X.X/10
📏 Facial Proportions: X.X/10
🦴 Jawline Definition: X.X/10
👃 Nose: X.X/10
👄 Lips: X.X/10
👀 Eye Area: X.X/10

🧔 Potential: Low / Medium / High

Strengths:
✅ item
✅ item

Weaknesses:
⚠ item
⚠ item

Recommendations:
• item
• item
• item

Keep the tone constructive and neutral.
Do not insult the person.
"""


def analyze_face(image_path):
    metrics = calculate_face_metrics(image_path)

    if metrics is None:
        return (
            "❌ Не удалось обнаружить лицо.\n\n"
            "Попробуйте загрузить фото:\n"
            "• анфас\n"
            "• хорошее освещение\n"
            "• один человек в кадре"
        )

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(
        build_prompt(metrics)
    )

    return response.text
