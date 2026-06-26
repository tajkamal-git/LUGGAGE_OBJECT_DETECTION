import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-this-in-production")

    # Hard cap per uploaded image. Vercel Python functions also impose their
    # own body-size ceiling at the platform level; this is the app-level guard.
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024   # 12 MB per request

    DEFAULT_CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

    SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

    # JPEG quality used when re-encoding the annotated result for the
    # base64 round-trip back to the browser (no disk writes — see README
    # "Why no file storage" for why this matters on serverless hosts).
    RESULT_JPEG_QUALITY = 85
