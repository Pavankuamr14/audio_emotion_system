# Audio Emotion System - Installation Guide

If you're encountering dependency conflicts or errors when running the system, follow these steps to fix the issues.

## Option 1: Use the Environment Fixer Script (Recommended)

1. Activate your virtual environment:
   ```
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. Run the environment fixer script:
   ```
   python fix_environment.py
   ```

3. After successful installation, run the application:
   ```
   python main.py
   ```

## Option 2: Manual Installation

If the fixer script doesn't work, you can perform the steps manually:

1. Activate your virtual environment:
   ```
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. Uninstall potentially conflicting packages:
   ```
   pip uninstall -y tensorflow tensorflow-cpu keras numpy pandas scipy librosa pygame pydub sounddevice playsound simpleaudio
   ```

3. Install packages in the correct order:
   ```
   pip install -r requirements_correct.txt
   ```

## Option 3: Minimal Installation (No TensorFlow)

If you just want the basic functionality without TensorFlow (uses random emotion predictions):

1. Activate your virtual environment:
   ```
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. Uninstall all packages:
   ```
   pip uninstall -y tensorflow tensorflow-cpu keras numpy pandas scipy librosa pygame pydub sounddevice playsound simpleaudio
   ```

3. Install minimal requirements:
   ```
   pip install -r requirements_minimal.txt
   ```

## Troubleshooting Common Issues

### Pygame Audio Not Working

If Pygame audio is not working:

1. Make sure your computer has working audio output
2. Check if you have the Microsoft Visual C++ Redistributable installed
3. Try another set of speakers or headphones

### Microphone Not Working

If your microphone isn't being detected or working:

1. Check your computer's sound settings to ensure the microphone is enabled
2. Try selecting different device numbers when prompted
3. Ensure you're not muted and speak at a reasonable volume

### Import Errors

If you still see import errors:

1. Try creating a completely new virtual environment:
   ```
   python -m venv new_venv
   new_venv\Scripts\activate
   pip install -r requirements_minimal.txt
   ```

2. Run the simplified version of the application that avoids using TensorFlow 