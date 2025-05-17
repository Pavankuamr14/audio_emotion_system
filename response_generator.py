import random
import logging
import tempfile
import os
import time
import win32com.client
import pythoncom
import subprocess
import winsound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowsSpeaker:
    """Windows speech synthesizer using SAPI directly."""
    
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.speaker = None
        self._initialize_speaker()
        
    def _initialize_speaker(self):
        """Initialize the Windows SAPI speaker."""
        try:
            pythoncom.CoInitialize()
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Configure voice settings
            self.speaker.Rate = 0  # Normal speed (-10 to 10)
            self.speaker.Volume = 100  # Full volume (0 to 100)
            
            # Try to select a female voice if available
            voices = self.speaker.GetVoices()
            for i in range(voices.Count):
                voice = voices.Item(i)
                if "female" in voice.GetDescription().lower():
                    self.speaker.Voice = voice
                    break
            
            if not self.quiet:
                print("Windows speech system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Windows SAPI: {e}")
            self.speaker = None
            return False
    
    def speak(self, text):
        """Speak the given text."""
        if not text:
            return
        
        # Always show the text
        if not self.quiet:
            print("\nSystem Response:")
            print("-" * 40)
            print(text)
            print("-" * 40)
        
        success = False
        
        try:
            # Ensure COM is initialized in this thread
            pythoncom.CoInitialize()
            
            # Reinitialize speaker if needed
            if not self.speaker:
                if not self._initialize_speaker():
                    raise Exception("Could not initialize speaker")
            
            if not self.quiet:
                print("\nSpeaking response...")
            
            # Speak synchronously to ensure completion
            self.speaker.Speak(text, 0)  # 0 means synchronous
            success = True
            
        except Exception as e:
            logger.error(f"Speech failed: {e}")
            # Provide audio feedback on failure
            try:
                winsound.Beep(1000, 100)
                time.sleep(0.1)
                winsound.Beep(800, 100)
            except:
                pass
            
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            
        return success

class EnhancedResponseGenerator:
    """Generate natural and context-aware emotional responses."""
    
    def __init__(self):
        self.response_templates = {
            'neutral': [
                "I notice that your voice sounds calm and balanced. {confidence}",
                "Your tone suggests a steady, composed state of mind. {confidence}",
                "I'm picking up a neutral emotional quality in your voice. {confidence}",
                "Your voice carries a sense of equilibrium. {confidence}",
                "There's a measured, even quality to your voice right now. {confidence}"
            ],
            'happy': [
                "I can hear the joy in your voice! {confidence}",
                "Your voice has such a wonderful, upbeat quality to it! {confidence}",
                "That happiness in your voice is really shining through! {confidence}",
                "Your cheerful tone is absolutely delightful to hear! {confidence}",
                "There's such a positive energy radiating through your voice! {confidence}"
            ],
            'sad': [
                "I'm hearing some sadness in your voice. Remember, it's okay to feel this way. {confidence}",
                "Your voice suggests you might be feeling down. Would you like to talk about it? {confidence}",
                "I sense some heaviness in your tone. Know that these feelings are valid. {confidence}",
                "There's a touch of melancholy in your voice. I'm here to listen. {confidence}",
                "Your voice carries some sadness. Take all the time you need. {confidence}"
            ],
            'angry': [
                "I can hear the intensity in your voice. Would you like to talk about what's bothering you? {confidence}",
                "There's a strong emotion coming through in your voice. Let's work through this together. {confidence}",
                "Your voice suggests you're feeling frustrated. That's completely understandable. {confidence}",
                "I'm picking up on some anger in your tone. Would you like to discuss what's on your mind? {confidence}",
                "There's passionate energy in your voice. I'm here to listen and understand. {confidence}"
            ]
        }
        
        self.confidence_phrases = {
            'high': "I'm quite confident about sensing this in your voice.",
            'medium': "I believe this is what I'm hearing in your voice.",
            'low': "This is what I'm perceiving, though I'm not entirely certain."
        }
    
    def _get_confidence_phrase(self, confidence):
        """Get appropriate confidence phrase based on confidence level."""
        if confidence > 0.8:
            return self.confidence_phrases['high']
        elif confidence > 0.5:
            return self.confidence_phrases['medium']
        else:
            return self.confidence_phrases['low']
    
    def generate_response(self, emotion_result):
        """Generate a natural, context-aware response based on detected emotion."""
        try:
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            
            # Get base response template
            templates = self.response_templates.get(emotion, self.response_templates['neutral'])
            base_response = random.choice(templates)
            
            # Add confidence information
            confidence_phrase = self._get_confidence_phrase(confidence)
            response = base_response.format(confidence=confidence_phrase)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm processing what I hear in your voice."