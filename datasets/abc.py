import joblib

import app  # noqa: F401 — ensure project root on sys.path
from app.paths import MODEL_PATH

model = joblib.load(MODEL_PATH)

print(model.feature_names_in_)