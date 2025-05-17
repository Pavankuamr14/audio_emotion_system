"""
Fix environment script to reinstall packages in the correct order with compatible versions.
Run this script from your virtual environment.
"""
import os
import sys
import subprocess
import time
import pyttsx3
import random
import logging
from gtts import gTTS
from playsound import playsound
import tempfile
from response_generator import EnhancedResponseGenerator, WindowsSpeaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a command and print its output"""
    print(f"\n[Running] {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(f"[Error] {result.stderr}")
    return result.returncode == 0

def main():
    print("\n=== Audio Emotion System Environment Fixer ===")
    print("This script will fix package compatibility issues by reinstalling packages in the correct order.")
    
    # Make sure we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n[ERROR] You are not running in a virtual environment!")
        print("Please activate your virtual environment first with:")
        print("  venv\\Scripts\\activate  (on Windows)")
        print("  source venv/bin/activate (on macOS/Linux)")
        return False

    # Uninstall conflicting packages
    print("\n=== Uninstalling potentially conflicting packages ===")
    packages_to_uninstall = [
        "tensorflow", "tensorflow-cpu", "keras", 
        "numpy", "pandas", "scipy", "librosa",
        "pygame", "pydub", "sounddevice", "playsound", "simpleaudio"
    ]
    
    for package in packages_to_uninstall:
        run_command(f"{sys.executable} -m pip uninstall -y {package}")
    
    # Clean pip cache to be safe
    run_command(f"{sys.executable} -m pip cache purge")
    
    # Install packages in the correct order from our fixed requirements file
    print("\n=== Installing packages in the correct order ===")
    success = run_command(f"{sys.executable} -m pip install -r requirements_correct.txt")
    
    if success:
        print("\n=== Environment setup completed successfully! ===")
        print("You can now run main.py to start the audio emotion recognition system.")
    else:
        print("\n[ERROR] There were problems during package installation.")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
    print("\nPress Enter to exit...")
    input()

# Test speech synthesis
engine = pyttsx3.init()
engine.say("Test pyttsx3")
engine.runAndWait()

class SpeechSynthesizer:
    def __init__(self, quiet=False):
        """Initialize speech synthesizer."""
        self.quiet = quiet
        self.temp_dir = tempfile.gettempdir()
        if not quiet:
            print("Speech system initialized")

    def speak(self, text):
        """Convert text to speech and play it."""
        if not text:
            return

        try:
            # Create temporary file path
            temp_file = os.path.join(self.temp_dir, 'temp_speech.mp3')
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            
            # Play the speech
            if not self.quiet:
                print("\nPlaying response...")
            playsound(temp_file)
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            if not self.quiet:
                print(f"Could not speak the response: {e}")

class ResponseGenerator:
    def __init__(self):
        """Initialize response templates."""
        self.responses = {
            'neutral': [
                "I sense that you're feeling balanced and composed.",
                "You seem to be in a calm state of mind.",
                "Your tone suggests a neutral, steady mood."
            ],
            'happy': [
                "Your joy is contagious! Keep spreading that positive energy!",
                "I can hear the happiness in your voice. That's wonderful!",
                "Your cheerful tone brightens the conversation!"
            ],
            'sad': [
                "I hear sadness in your voice. Remember, it's okay to feel this way.",
                "You sound down. Would you like to talk about what's bothering you?",
                "I sense that you're feeling low. Take your time to process your emotions."
            ],
            'angry': [
                "I can hear that you're frustrated. Taking deep breaths might help.",
                "Your voice suggests anger. It's natural to feel this way sometimes.",
                "I sense strong emotions in your voice. Would you like to talk about it?"
            ]
        }
        
    def generate_response(self, emotion_result):
        """Generate an appropriate response based on detected emotion."""
        try:
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            
            templates = self.responses.get(emotion, self.responses['neutral'])
            response_text = random.choice(templates)
            
            if confidence > 0.7:
                response_text += f" I'm quite confident about sensing this emotion."
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response text: {e}")
            return "I'm having trouble understanding the emotion right now."

# Initialize the speech synthesizer and response generator
speech_synth = WindowsSpeaker(quiet=False)
response_gen = EnhancedResponseGenerator() 