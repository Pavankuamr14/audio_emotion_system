"""
An improved rule-based emotion recognizer that analyzes audio features to detect emotions.
"""
import numpy as np
import logging
from typing import Dict, Any
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionRecognizer:
    """
    A rule-based emotion classifier that analyzes audio features to detect emotions.
    """
    
    def __init__(self, quiet=False):
        """Initialize the emotion recognizer with improved thresholds."""
        self.emotions = ['neutral', 'happy', 'sad', 'angry']
        self.feature_thresholds = {
            'energy': {
                'low': 0.2,    # Normalized energy thresholds
                'high': 0.6
            },
            'zero_crossing_rate': {
                'low': 0.1,
                'high': 0.3
            },
            'spectral_centroid': {
                'low': 1000,
                'high': 2000
            },
            'spectral_rolloff': {
                'low': 3000,
                'high': 6000
            }
        }
        logger.info("Rule-based emotion recognizer initialized with enhanced feature analysis")
        if not quiet:
            print("Using enhanced emotion recognition system")
    
    def _analyze_features(self, features):
        """
        Analyze audio features to determine emotion probabilities with improved accuracy.
        
        Args:
            features: Dictionary containing audio features
            
        Returns:
            Dictionary with emotion probabilities
        """
        # Initialize probabilities
        probabilities = {emotion: 0.0 for emotion in self.emotions}
        
        # Extract all features
        energy = features.get('energy', 0.0)
        zcr = features.get('zero_crossing_rate', 0.0)
        mfcc = features.get('mfcc', np.zeros(13))
        delta_mfcc = features.get('delta_mfcc', np.zeros(13))
        spectral_centroid = features.get('spectral_centroid', 0.0)
        spectral_rolloff = features.get('spectral_rolloff', 0.0)
        
        # Analyze energy (intensity of speech)
        if energy > self.feature_thresholds['energy']['high']:
            # High energy suggests angry or happy
            probabilities['angry'] += 0.4
            probabilities['happy'] += 0.3
        elif energy < self.feature_thresholds['energy']['low']:
            # Low energy suggests sad or neutral
            probabilities['sad'] += 0.4
            probabilities['neutral'] += 0.3
        
        # Analyze zero crossing rate (pitch variations)
        if zcr > self.feature_thresholds['zero_crossing_rate']['high']:
            # High ZCR suggests happy or angry
            probabilities['happy'] += 0.3
            probabilities['angry'] += 0.2
        elif zcr < self.feature_thresholds['zero_crossing_rate']['low']:
            # Low ZCR suggests sad or neutral
            probabilities['sad'] += 0.3
            probabilities['neutral'] += 0.2
        
        # Analyze spectral centroid (brightness of sound)
        if spectral_centroid > self.feature_thresholds['spectral_centroid']['high']:
            # Higher centroid suggests happy
            probabilities['happy'] += 0.3
        elif spectral_centroid < self.feature_thresholds['spectral_centroid']['low']:
            # Lower centroid suggests sad
            probabilities['sad'] += 0.3
        
        # Analyze spectral rolloff (amount of high frequencies)
        if spectral_rolloff > self.feature_thresholds['spectral_rolloff']['high']:
            # More high frequencies suggest angry or happy
            probabilities['angry'] += 0.2
            probabilities['happy'] += 0.2
        elif spectral_rolloff < self.feature_thresholds['spectral_rolloff']['low']:
            # Fewer high frequencies suggest sad or neutral
            probabilities['sad'] += 0.2
            probabilities['neutral'] += 0.2
        
        # Analyze MFCC dynamics (changes in vocal characteristics)
        if np.mean(np.abs(delta_mfcc)) > 0.1:
            # High MFCC variation suggests emotional speech (angry/happy)
            probabilities['angry'] += 0.2
            probabilities['happy'] += 0.2
        else:
            # Low MFCC variation suggests calmer speech (neutral/sad)
            probabilities['neutral'] += 0.2
            probabilities['sad'] += 0.1
        
        # Normalize probabilities to sum to 1
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities
    
    def predict_emotion(self, features):
        """
        Predict emotion from audio features with confidence scoring.
        
        Args:
            features: Audio features extracted from the signal
            
        Returns:
            Dictionary with predicted emotion, confidence, and all probabilities
        """
        try:
            # Get emotion probabilities
            probabilities = self._analyze_features(features)
            
            # Find the emotion with highest probability
            top_emotion = max(probabilities.items(), key=lambda x: x[1])
            emotion, confidence = top_emotion
            
            # Get second highest probability for confidence adjustment
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) > 1:
                second_prob = sorted_probs[1][1]
                # Adjust confidence based on difference with second highest
                confidence_margin = confidence - second_prob
                # Scale confidence to reflect certainty of prediction
                confidence = min(1.0, confidence + confidence_margin)
            
            logger.info(f"Predicted emotion: {emotion} with confidence {confidence:.2f}")
            logger.info(f"Probability distribution: {', '.join(f'{k}: {v:.2f}' for k, v in probabilities.items())}")
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'probabilities': {emotion: 0.25 for emotion in self.emotions}
            }

def extract_features(audio_data, sample_rate):
    """
    Extract audio features for emotion recognition with improved feature extraction.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary of audio features
    """
    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.ndim == 0 or audio_data.size == 0:
            logger.warning("Empty or invalid audio data received for feature extraction.")
            return {
                'mfcc': np.zeros(13, dtype=np.float32),
                'energy': 0.0,
                'zero_crossing_rate': 0.0,
                'audio_data': np.zeros(16000, dtype=np.float32),
                'sample_rate': sample_rate
            }

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        
        # Normalize audio data
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        features = {}
        
        # Extract MFCCs with improved parameters
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate, 
            n_mfcc=13,
            n_fft=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.010 * sample_rate)  # 10ms hop
        )
        features['mfcc'] = np.mean(mfccs, axis=1) if mfccs.size > 0 else np.zeros(13, dtype=np.float32)
        
        # Calculate energy with noise floor consideration
        energy = np.sum(audio_data ** 2) / len(audio_data) if len(audio_data) > 0 else 0.0
        # Apply a small noise floor threshold (0.001) to prevent very low energy values
        energy = max(energy, 0.001)
        features['energy'] = energy
        
        # Extract zero crossing rate with improved parameters
        zero_crossings = librosa.feature.zero_crossing_rate(
            audio_data,
            frame_length=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.010 * sample_rate)     # 10ms hop
        )
        features['zero_crossing_rate'] = np.mean(zero_crossings) if zero_crossings.size > 0 else 0.0
        
        # Add raw audio data and sample rate for advanced feature extraction
        features['audio_data'] = audio_data
        features['sample_rate'] = sample_rate
        
        # Log feature values for debugging
        logger.info(f"Extracted features - Energy: {features['energy']:.4f}, ZCR: {features['zero_crossing_rate']:.4f}")
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features (SR: {sample_rate}): {e}")
        return {
            'mfcc': np.zeros(13, dtype=np.float32),
            'energy': 0.0,
            'zero_crossing_rate': 0.0,
            'audio_data': np.zeros(16000, dtype=np.float32),
            'sample_rate': sample_rate
        }