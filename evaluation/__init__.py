"""
Evaluation module for the UAM Literature Review RAG System.
Contains comprehensive evaluation tools and metrics.
"""

from .evaluator import UAMResearchEvaluator
from .metrics import RAGEvaluationSystem, EvaluationResult
from .emergency_eval import emergency_evaluation

__all__ = [
    'UAMResearchEvaluator',
    'RAGEvaluationSystem', 
    'EvaluationResult',
    'emergency_evaluation'
]