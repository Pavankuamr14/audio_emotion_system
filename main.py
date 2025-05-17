import argparse
import sys
import time
import logging
import sounddevice as sd

from audio_utils import AudioRecorder, extract_features, list_audio_devices, detect_silence
from emotion_recognizer import EmotionRecognizer
from response_generator import EnhancedResponseGenerator, RobustSpeaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Audio Emotion Recognition System')
    parser.add_argument('--continuous', action='store_true',
                      help='Run in continuous mode (processes audio after each stop)')
    parser.add_argument('--device', type=int,
                      help='Audio input device index (e.g., 3)')
    parser.add_argument('--no-speech', action='store_true',
                      help='Disable spoken responses')
    parser.add_argument('--quiet', action='store_true',
                      help='Reduce terminal output (only show essential information)')
    return parser.parse_args()

def select_audio_device(quiet=False):
    """Let user select an audio input device."""
    input_devices = list_audio_devices(quiet=quiet)
    
    if not input_devices:
        print("\nNo audio input devices found!")
        return None
        
    try:
        device_str = input("\nEnter the number of the device you want to use (or press Enter for default): ").strip()
        if device_str:
            device_index = int(device_str)
            if device_index in input_devices:
                logger.info(f"User selected device: {device_index}")
                return device_index
            else:
                print(f"Invalid device number: {device_index}")
                return None
    except ValueError:
        print("Please enter a valid number")
        return None
    
    return None

def record_and_analyze(device_index=None, speak_response=True, quiet=False):
    """Record audio indefinitely until Ctrl+C, then analyze emotions and speak response."""
    recorder = None
    speech_synth = None
    if speak_response:
        speech_synth = RobustSpeaker(quiet=quiet)

    try:
        recorder = AudioRecorder(device_index=device_index, quiet=quiet)
        recognizer = EmotionRecognizer(quiet=quiet)
        response_gen = EnhancedResponseGenerator()

        if recorder.device_index is None:
            logger.error("AudioRecorder could not be initialized with a valid device.")
            print("Failed to initialize audio recorder. Please check microphone or select a valid device.")
            return False
        
        if not quiet:
            print("\nPreparing to record... (Press Ctrl+C to cancel at any time)")
        
        # Short pause before starting, interruptible
        try:
            time.sleep(1) 
        except KeyboardInterrupt:
            print("\nRecording preparation cancelled.")
            return False

        print("\nRecording... Speak now! (Press Ctrl+C to stop and analyze)")
        recorder.start_recording()
        
        try:
            while True: # Record indefinitely
                time.sleep(0.1) # Keep the loop responsive
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        
        if not quiet:
            print("\nProcessing recorded audio...")
        audio_data = recorder.stop_recording()
        sample_rate = recorder.sample_rate
        recorder = None 
        
        if audio_data is None or len(audio_data) == 0:
            print("\nNo audio was recorded or retrieved. Please try again.")
            return False
        
        # Check if audio is just silence
        if detect_silence(audio_data):
            print("\nNo speech detected. Please speak clearly during recording.")
            
            # Provide a voice response for silence
            if speak_response and speech_synth:
                silence_message = "I didn't hear any speech. Please speak clearly when recording."
                print("\nSystem Response:")
                if not quiet:
                    print("-" * 40)
                print(silence_message)
                if not quiet:
                    print("-" * 40)
                
                if not quiet:
                    print("\nSystem Response (Audio): Speaking...")
                speech_synth.speak(silence_message)
            
            return True  # Return True so the session doesn't end immediately
            
        features = extract_features(audio_data, sample_rate=sample_rate)
        emotion_result = recognizer.predict_emotion(features)
        
        print("\nAnalysis Results:")
        if not quiet:
            print("-" * 40)
        print(f"Detected Emotion: {emotion_result['emotion']} (Confidence: {emotion_result['confidence']:.1%})")
        
        if not quiet:
            print("\nEmotion Probabilities:")
            for emotion, prob in emotion_result['probabilities'].items():
                print(f"  {emotion}: {prob:.1%}")
            
        response_text = response_gen.generate_response(emotion_result)
        print("\nSystem Response:")
        if not quiet:
            print("-" * 40)
        print(response_text)
        if not quiet:
            print("-" * 40)

        if speak_response and speech_synth:
            if not quiet:
                print("\nSystem Response (Audio): Speaking...")
            speech_synth.speak(response_text)
        elif speak_response:
            print("\n(Audio response disabled as speech synthesizer could not be initialized)")
        
        return True
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        if recorder and recorder.is_recording:
            print("Stopping active recording...")
            recorder.stop_recording()
        return False
    except Exception as e:
        logger.error(f"Error during recording and analysis: {e}")
        print(f"An error occurred: {e}")
        if recorder and recorder.is_recording:
            recorder.stop_recording()
        return False

def main():
    """Main function."""
    try:
        args = parse_arguments()
        # Set logging level based on quiet option
        if args.quiet:
            logging.basicConfig(level=logging.WARNING, force=True)
        print("\nAudio Emotion Recognition System")
        if not args.quiet:
            print("=" * 50)
        selected_device_index = args.device
        if selected_device_index is None:
            selected_device_index = select_audio_device(quiet=args.quiet)
        if not args.quiet:
            print("\nTips for best results:")
            print("- Speak clearly and at a normal volume")
            print("- Keep background noise to a minimum")
            print("- Recording will continue until you press Ctrl+C")
            if args.no_speech:
                print("- Spoken responses are DISABLED.")
        while True:
            try:
                success = record_and_analyze(selected_device_index, speak_response=not args.no_speech, quiet=args.quiet)
                if not args.continuous:
                    if not success:
                        print("Session ended.")
                    break # Exit after one session if not continuous
                if not success:
                    print("An error occurred in the previous session, or it was cancelled.")
                response = input("\nPress Enter to start another session or 'q' to quit: ").strip().lower()
                if response == 'q':
                    break
            except KeyboardInterrupt:
                print("\n[INFO] Program interrupted by user (Ctrl+C). Exiting...")
                sys.exit(0)
        print("\nThank you for using the Audio Emotion Recognition System!")
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user (Ctrl+C). Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"A critical error occurred in main: {e}")
        print(f"A critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user (Ctrl+C). Exiting...")
        sys.exit(0)