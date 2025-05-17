import random
import logging
import tempfile
import os
import time
import threading
import pyttsx3
from gtts import gTTS
import pygame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        if confidence > 0.8:
            return self.confidence_phrases['high']
        elif confidence > 0.5:
            return self.confidence_phrases['medium']
        else:
            return self.confidence_phrases['low']
    def generate_response(self, emotion_result):
        try:
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            templates = self.response_templates.get(emotion, self.response_templates['neutral'])
            base_response = random.choice(templates)
            confidence_phrase = self._get_confidence_phrase(confidence)
            response = base_response.format(confidence=confidence_phrase)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm processing what I hear in your voice."

class RobustSpeaker:
    """Robust TTS: pyttsx3 -> gTTS fallback with logging and timeout. Uses pygame for playback."""
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.pyttsx3_engine = None
        self._init_pyttsx3()
        self.temp_dir = tempfile.gettempdir()
        pygame.mixer.init()
    def _init_pyttsx3(self):
        try:
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', 150)
            self.pyttsx3_engine.setProperty('volume', 1.0)
            voices = self.pyttsx3_engine.getProperty('voices')
            for v in voices:
                if 'female' in v.name.lower():
                    self.pyttsx3_engine.setProperty('voice', v.id)
                    break
            if not self.quiet:
                print("pyttsx3 speech system initialized")
        except Exception as e:
            logger.warning(f"pyttsx3 initialization failed: {e}")
            self.pyttsx3_engine = None
    def speak(self, text):
        if not text:
            return
        success = False
        # Try pyttsx3 with timeout
        if self.pyttsx3_engine:
            logger.info("Trying pyttsx3 for TTS...")
            def tts_job():
                try:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                except Exception as e:
                    logger.error(f"pyttsx3 speech failed: {e}")
            t = threading.Thread(target=tts_job)
            t.daemon = True  # Make thread daemon so it won't block exit
            t.start()
            t.join(timeout=8)  # 8 seconds max
            if t.is_alive():
                logger.error("pyttsx3 TTS timed out.")
            else:
                success = True
        # Try gTTS with pygame for playback
        if not success:
            logger.info("Trying gTTS for TTS fallback with pygame...")
            temp_file = os.path.join(self.temp_dir, f'temp_speech_{int(time.time())}.mp3')
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_file)
                print(f"[INFO] Playing audio file: {temp_file}")
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                try:
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n[INFO] Playback interrupted by user (Ctrl+C). Stopping audio...")
                    pygame.mixer.music.stop()
                    raise
                # Delay and retry for file deletion
                for _ in range(5):
                    try:
                        os.remove(temp_file)
                        break
                    except PermissionError:
                        time.sleep(0.2)
                else:
                    logger.warning(f"Could not delete temp file: {temp_file}")
                success = True
            except Exception as e:
                logger.error(f"gTTS speech failed: {e}")
        if not success:
            logger.error("All TTS engines failed. No audio output possible.")
            if not self.quiet:
                print("[ERROR] All TTS engines failed. No audio output possible.")