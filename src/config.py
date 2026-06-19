import os

# --- Model Configurations ---
# Paths to model weights (put your customized/pre-trained models in the models/ folder)
YOLO_MODEL_PATH = "yolov8n.pt"  # Person detection, tracking & poses
PPE_MODEL_PATH = "yolov8n.pt"   # Custom safety gear model (e.g. helmets/glasses)

# --- Confidence & Detection Thresholds ---
PERSON_CONF_THRESHOLD = 0.4
PPE_CONF_THRESHOLD = 0.5

# --- Privacy Settings (Person 3) ---
PRIVACY_BLUR_KERNEL_SIZE = (51, 51)  # Higher values = stronger blur
PRIVACY_FACE_PADDING = 10           # Bounding box padding in pixels for faces
BLUR_TATOOS = False                 # Toggle if tattoo/skin segmentation is active

# --- Quality & Safety Settings (Person 5) ---
BLUR_LAPLACIAN_THRESHOLD = 20.0     # Under this value, image is flagged as blurry/smudged
SMOKE_CONF_THRESHOLD = 0.5          # Threshold for environmental smoke detection
FALL_ANGLE_THRESHOLD = 50           # Angle (degrees) of spine relative to vertical (e.g. > 60 = horizontal/lying)
FALL_CONFIRMATION_FRAMES = 2        # Number of consecutive frames required to confirm a fall
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35 # Minimum confidence for pose keypoints to be used
FALL_ASPECT_RATIO_THRESHOLD = 1.75   # Width/height ratio threshold for aspect ratio fallback detection

# --- Re-Identification settings (Person 2) ---
REID_COSINE_SIMILARITY_THRESHOLD = 0.7  # Above this, it's considered the same person

# --- Privacy & Compliance Settings (GDPR) ---
PRIVACY_ENABLED = True                  # Enable privacy pipeline
PRIVACY_SALT_ROTATION_HOURS = 24        # How often to rotate ID anonymization salt
PRIVACY_AGGREGATION_WINDOW_MINUTES = 60 # Batch events into hourly windows (not point-in-time)
PRIVACY_RETENTION_DAYS = 90             # Keep aggregated data for 90 days, then purge
PRIVACY_AUTO_PURGE = True               # Automatically delete expired data
PRIVACY_ALLOW_FRAME_EXPORT = False      # NEVER export raw frames (privacy critical)
PRIVACY_AUDIT_LOG = True                # Log all privacy operations (for compliance)

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
