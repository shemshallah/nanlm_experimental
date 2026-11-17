"""Default, causal M3 inference: signed-pair/triangle successor memory."""
from .predictor import M3DefaultPredictor, M3R21Predictor, DEFAULT_MODE, MODES

__all__ = ["M3DefaultPredictor", "M3R21Predictor", "DEFAULT_MODE", "MODES"]
