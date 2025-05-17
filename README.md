# Audio Emotion Recognition System

A lightweight command-line tool for real-time emotion recognition from audio input. The system records audio from your microphone, analyzes the emotional content, and provides appropriate text and spoken responses.

## Features

- Real-time audio recording from microphone (stop with Ctrl+C)
- Emotion detection from voice
- Text responses based on detected emotions
- **Spoken audio responses** using Text-to-Speech (TTS)
- Works on CPU (no GPU required)
- Minimal dependencies
- Cross-platform support (Windows, macOS, Linux)

## Requirements

- Python 3.8 or higher
- Working microphone (built-in or external)
- Working speakers or headphones for audio responses
- pip package manager
- **Text-to-Speech Engine**: `pyttsx3` attempts to use native TTS engines:
    - Windows: SAPI5
    - macOS: NSSpeechSynthesizer (VoiceOver)
    - Linux: eSpeak (may require `sudo apt-get install espeak` or similar for your distribution if not present)

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `espeak` is needed on Linux and not installed, the `pyttsx3` initialization might fail. Install it using your package manager (e.g., `sudo apt-get install espeak`).

## Usage

1. Make sure your microphone is working and properly connected.
2. Ensure your speakers/headphones are working to hear the spoken response.
3. Run the program:
   ```bash
   python main.py
   ```
4. The system will prompt you to select an audio input device.
5. Recording will start and continue indefinitely. Press **Ctrl+C** to stop recording and process the audio.
6. The system will print the text response and then speak it.
7. Optional arguments:
   - `--device N`: Specify the audio input device index `N` directly (e.g., `--device 3`).
   - `--continuous`: After processing, ask to start another session. Without this, the program exits after one recording.
   - `--no-speech`: Disable spoken audio responses (only text output).
   ```bash
   python main.py --device 3 --continuous --no-speech
   ```

**Important Note on Recording Duration:** Recording continues until you press Ctrl+C. For very long recording sessions, the program will consume more memory as it buffers all audio data.

## Tips for Best Results

1. Use a good quality microphone.
2. Speak clearly and at a normal volume.
3. Minimize background noise.
4. Position yourself close to the microphone.

## Troubleshooting

1. No audio detected:
   - Check if your microphone is properly connected and selected.
   - Make sure it's not muted in system settings.
   - Try speaking louder or moving closer to the microphone.

2. No spoken response / TTS errors:
   - Ensure `pyttsx3` installed correctly.
   - On Linux, you might need to install `espeak` (`sudo apt-get install espeak`).
   - Check your system's audio output and volume levels.
   - Run with `--no-speech` to verify the rest of the system works.

3. Poor recognition:
   - Ensure minimal background noise.
   - Speak clearly and at a consistent volume.
   - Try adjusting your distance from the microphone.
   - Note: The included emotion model is a non-trained dummy for structure; for real accuracy, a trained model is needed.

## License

MIT License - Feel free to use and modify for your needs. 