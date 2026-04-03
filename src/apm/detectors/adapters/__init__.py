"""Detector adapter implementations."""

from apm.detectors.adapters.aigc_detector_env3 import AIGCDetectorEnv3
from apm.detectors.adapters.aigc_detector_env3short import AIGCDetectorEnv3Short
from apm.detectors.adapters.detectgpt_light import DetectGptLightDetector
from apm.detectors.adapters.gltr_gpt2_small import GLTRGpt2SmallDetector
from apm.detectors.adapters.radar_vicuna_7b import RadarVicuna7BDetector
from apm.detectors.adapters.seqxgpt import SeqXGPTDetector
from apm.detectors.adapters.synthid_text import SynthIDTextDetector

__all__ = [
    "AIGCDetectorEnv3",
    "AIGCDetectorEnv3Short",
    "GLTRGpt2SmallDetector",
    "DetectGptLightDetector",
    "RadarVicuna7BDetector",
    "SeqXGPTDetector",
    "SynthIDTextDetector",
]
