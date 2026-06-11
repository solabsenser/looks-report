"""Backward-compatible wrapper for the face analysis implementation.

The application imports ``face_analyzer`` directly. This module is kept only for
older local scripts that still import ``analyzer``.
"""

from face_analyzer import ANALYZER_BACKEND, analyze_face

__all__ = ["ANALYZER_BACKEND", "analyze_face"]
