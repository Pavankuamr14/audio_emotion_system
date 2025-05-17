"""
A simplified emotion recognizer that doesn't rely on TensorFlow.
This is a dummy implementation that returns random predictions but maintains the same API.
"""
import random
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionRecognizer:
    """
    A simplified emotion classifier that returns random predictions.
    This avoids TensorFlow dependency issues while maintaining the same API.
    """
    
    def __init__(self, quiet=False):
        """Initialize the dummy emotion recognizer."""
        self.emotions = ['neutral', 'happy', 'sad', 'angry']
        logger.info("Simple emotion recognizer initialized (random prediction mode)")
        logger.warning("This is a placeholder model that returns random emotion predictions")
        if not quiet:
            print("NOTE: Using simplified emotion recognizer (random predictions)")
        
    def predict_emotion(self, features):
        """
        Predict emotion from audio features.
        This is a dummy implementation that returns random predictions.
        
        Args:
            features: Audio features extracted from the signal
            
        Returns:
            Dictionary with predicted emotion, confidence, and all probabilities
        """
        # Generate random probabilities and normalize them to sum to 1
        raw_probs = np.random.random(len(self.emotions))
        probabilities = raw_probs / raw_probs.sum()
        
        # Create dictionary of emotions and their probabilities
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, probabilities)}
        
        # Find the emotion with the highest probability
        top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
        emotion, confidence = top_emotion
        
        logger.info(f"Predicted emotion: {emotion} with confidence {confidence:.2f}")
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': emotion_probs
        } 