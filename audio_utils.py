# import numpy as np
# import sounddevice as sd
# import librosa
# from scipy.io import wavfile
# import threading
# import queue
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def list_audio_devices(quiet=False):
#     """List all available audio input devices."""
#     devices = sd.query_devices()
#     input_devices = []
    
#     if not quiet:
#         print("\nAvailable Audio Input Devices:")
#         print("-" * 50)
#     else:
#         print("\nAvailable devices:", end=" ")
    
#     for i, device in enumerate(devices):
#         if device['max_input_channels'] > 0:  # If it's an input device
#             if not quiet:
#                 print(f"{i}: {device['name']} (Default SR: {int(device['default_samplerate'])} Hz)")
#             else:
#                 print(f"{i}:{device['name'].split('(')[0].strip()}", end=" ")
#             input_devices.append(i)
    
#     if quiet and input_devices:
#         print()  # End the line for compact output
            
#     return input_devices

# def get_default_device_index():
#     """Get the default input device index."""
#     try:
#         device_info = sd.query_devices(kind='input')
#         return device_info['index']
#     except Exception:
#         return None

# def detect_silence(audio_data, threshold=0.01):
#     """
#     Detect if audio is silence or contains speech.
    
#     Args:
#         audio_data: Numpy array of audio samples
#         threshold: RMS energy threshold to consider as non-silence
        
#     Returns:
#         True if silence, False if speech detected
#     """
#     if audio_data is None or len(audio_data) == 0:
#         return True
        
#     # Calculate RMS energy
#     rms = np.sqrt(np.mean(np.square(audio_data)))
#     logger.info(f"Audio RMS energy: {rms:.6f} (threshold: {threshold})")
    
#     # Check if below threshold
#     return rms < threshold

# class AudioRecorder:
#     def __init__(self, chunk_duration=1.0, device_index=None, quiet=False):
#         """Initialize audio recorder, automatically determining sample rate from device."""
#         self.chunk_duration = chunk_duration
#         self.is_recording = False
#         self.audio_buffer = []
#         self.sample_rate = 16000  # Default fallback sample rate
#         self.quiet = quiet

#         if device_index is None:
#             device_index = get_default_device_index()
#             if device_index is None:
#                 available_devices = sd.query_devices()
#                 input_device_indices = [i for i, d in enumerate(available_devices) if d['max_input_channels'] > 0]
#                 if input_device_indices:
#                     device_index = input_device_indices[0]
#                     logger.info(f"No default device, falling back to first available: {device_index}")
        
#         self.device_index = device_index

#         if self.device_index is not None:
#             try:
#                 device_info = sd.query_devices(self.device_index)
#                 self.sample_rate = int(device_info['default_samplerate'])
#                 logger.info(f"Using audio device: {self.device_index} - {device_info['name']} with sample rate: {self.sample_rate} Hz")
#                 if not quiet:
#                     print(f"Using device: {device_info['name']}")
#             except Exception as e:
#                 logger.error(f"Error querying device {self.device_index} for sample rate: {e}. Falling back to SR {self.sample_rate}Hz.")
#                 # Keep default sample_rate, device_index might still work or fail later
#         else:
#             logger.error("No input devices found or specified. AudioRecorder might not function.")

#         self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
#     def start_recording(self):
#         """Start recording audio from the microphone."""
#         if self.device_index is None:
#             logger.error("AudioRecorder: No audio input device available or selected.")
#             raise ValueError("No audio input device available or selected for recording.")
            
#         def audio_callback(indata, frames, time, status):
#             if status:
#                 logger.warning(f"Audio callback status: {status}")
#             audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 and indata.shape[1] > 0 else indata.flatten()
#             self.audio_buffer.append(audio_data.copy())
            
#         try:
#             logger.info(f"Attempting to start recording with device index: {self.device_index}, SR: {self.sample_rate} Hz")
#             self.stream = sd.InputStream(
#                 device=self.device_index,
#                 samplerate=self.sample_rate,
#                 channels=1,
#                 callback=audio_callback,
#                 blocksize=self.chunk_samples,
#                 dtype='float32'
#             )
#             self.audio_buffer = []
#             self.is_recording = True
#             self.stream.start()
#             logger.info(f"Started audio recording on device: {self.device_index}, SR: {self.sample_rate} Hz")
#         except Exception as e:
#             logger.error(f"Error starting audio recording on device {self.device_index} (SR: {self.sample_rate} Hz): {e}")
#             raise
            
#     def stop_recording(self):
#         """Stop recording and return the recorded audio."""
#         if hasattr(self, 'stream') and self.stream:
#             try:
#                 if self.stream.active:
#                     self.stream.stop()
#                 self.stream.close()
#             except Exception as e:
#                 logger.error(f"Error stopping/closing audio stream: {e}")
            
#             self.is_recording = False
#             logger.info("Stopped audio recording")
            
#             if self.audio_buffer:
#                 return np.concatenate(self.audio_buffer)
#             return np.array([], dtype=np.float32)
#         return np.array([], dtype=np.float32)

# def extract_features(audio_data, sample_rate):
#     """Extract audio features for emotion recognition. Requires sample_rate."""
#     try:
#         if not isinstance(audio_data, np.ndarray) or audio_data.ndim == 0 or audio_data.size == 0:
#             logger.warning("Empty or invalid audio data received for feature extraction.")
#             return {'mfcc': np.zeros(13, dtype=np.float32), 'energy': 0.0, 'zero_crossing_rate': 0.0}

#         if len(audio_data.shape) > 1:
#             audio_data = np.mean(audio_data, axis=1)
#         audio_data = audio_data.astype(np.float32)
        
#         features = {}
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
#         features['mfcc'] = np.mean(mfccs, axis=1) if mfccs.size > 0 else np.zeros(13, dtype=np.float32)
#         features['energy'] = np.sum(audio_data ** 2) / len(audio_data) if len(audio_data) > 0 else 0.0
#         zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
#         features['zero_crossing_rate'] = np.mean(zero_crossings) if zero_crossings.size > 0 else 0.0
        
#         return features
#     except Exception as e:
#         logger.error(f"Error extracting features (SR: {sample_rate}): {e}")
#         return {'mfcc': np.zeros(13, dtype=np.float32), 'energy': 0.0, 'zero_crossing_rate': 0.0} 

import numpy as np
import sounddevice as sd
import librosa
from scipy.io import wavfile
import threading
import queue
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_audio_devices(quiet=False):
    """List all available audio input devices."""
    devices = sd.query_devices()
    input_devices = []
    
    if not quiet:
        print("\nAvailable Audio Input Devices:")
        print("-" * 50)
    else:
        print("\nAvailable devices:", end=" ")
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # If it's an input device
            if not quiet:
                print(f"{i}: {device['name']} (Default SR: {int(device['default_samplerate'])} Hz)")
            else:
                print(f"{i}:{device['name'].split('(')[0].strip()}", end=" ")
            input_devices.append(i)
    
    if quiet and input_devices:
        print()  # End the line for compact output
            
    return input_devices

def get_default_device_index():
    """Get the default input device index."""
    try:
        device_info = sd.query_devices(kind='input')
        return device_info['index']
    except Exception:
        return None

def detect_silence(audio_data, threshold=0.005):  # Lowered threshold
    """
    Detect if audio is silence or contains speech.
    
    Args:
        audio_data: Numpy array of audio samples
        threshold: RMS energy threshold to consider as non-silence
        
    Returns:
        True if silence, False if speech detected
    """
    if audio_data is None or len(audio_data) == 0:
        return True
        
    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(np.square(audio_data)))
    logger.info(f"Audio RMS energy: {rms:.6f} (threshold: {threshold})")
    
    # Check if below threshold
    return rms < threshold

class AudioLevelIndicator:
    """Provides real-time audio level feedback."""
    def __init__(self, update_interval=0.1):
        self.update_interval = update_interval
        self.last_update = time.time()
        
    def update(self, audio_data):
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            normalized_rms = min(1.0, rms * 20)  # Scale up for better visibility
            bars = int(normalized_rms * 20)
            sys.stdout.write('\r')
            sys.stdout.write(f"Audio Level: {'|' * bars}{' ' * (20 - bars)} [{rms:.3f}]")
            sys.stdout.flush()
            self.last_update = current_time

class AudioRecorder:
    def __init__(self, chunk_duration=0.1, device_index=None, quiet=False):  # Reduced chunk duration for more responsive feedback
        """Initialize audio recorder with volume normalization and feedback."""
        self.chunk_duration = chunk_duration
        self.is_recording = False
        self.audio_buffer = []
        self.sample_rate = 16000  # Default fallback sample rate
        self.quiet = quiet
        self.level_indicator = None if quiet else AudioLevelIndicator()

        if device_index is None:
            device_index = get_default_device_index()
            if device_index is None:
                available_devices = sd.query_devices()
                input_device_indices = [i for i, d in enumerate(available_devices) if d['max_input_channels'] > 0]
                if input_device_indices:
                    device_index = input_device_indices[0]
                    logger.info(f"No default device, falling back to first available: {device_index}")
        
        self.device_index = device_index

        if self.device_index is not None:
            try:
                device_info = sd.query_devices(self.device_index)
                self.sample_rate = int(device_info['default_samplerate'])
                logger.info(f"Using audio device: {self.device_index} - {device_info['name']} with sample rate: {self.sample_rate} Hz")
                if not quiet:
                    print(f"Using device: {device_info['name']}")
            except Exception as e:
                logger.error(f"Error querying device {self.device_index} for sample rate: {e}. Falling back to SR {self.sample_rate}Hz.")

        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
    def start_recording(self):
        """Start recording audio with real-time level feedback."""
        if self.device_index is None:
            logger.error("AudioRecorder: No audio input device available or selected.")
            raise ValueError("No audio input device available or selected for recording.")
            
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to mono if needed and normalize
            audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 and indata.shape[1] > 0 else indata.flatten()
            
            # Normalize audio (make quieter sounds louder while preserving dynamics)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Apply gentle compression
            threshold = 0.3
            ratio = 0.6
            audio_data = np.where(
                np.abs(audio_data) > threshold,
                threshold + (np.abs(audio_data) - threshold) * ratio * np.sign(audio_data),
                audio_data
            )
            
            self.audio_buffer.append(audio_data.copy())
            
            # Update audio level indicator
            if self.level_indicator:
                self.level_indicator.update(audio_data)
            
        try:
            logger.info(f"Attempting to start recording with device index: {self.device_index}, SR: {self.sample_rate} Hz")
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=self.chunk_samples,
                dtype='float32'
            )
            self.audio_buffer = []
            self.is_recording = True
            self.stream.start()
            logger.info(f"Started audio recording on device: {self.device_index}, SR: {self.sample_rate} Hz")
            
            if not self.quiet:
                print("\nAudio level indicator:")
                print("=" * 50)
                
        except Exception as e:
            logger.error(f"Error starting audio recording on device {self.device_index} (SR: {self.sample_rate} Hz): {e}")
            raise
            
    def stop_recording(self):
        """Stop recording and return the processed audio."""
        if hasattr(self, 'stream') and self.stream:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping/closing audio stream: {e}")
            
            self.is_recording = False
            logger.info("Stopped audio recording")
            
            if self.audio_buffer:
                # Concatenate and process all recorded audio
                audio_data = np.concatenate(self.audio_buffer)
                
                # Normalize final audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
                
                # Clear the console line used for the level indicator
                if self.level_indicator:
                    sys.stdout.write('\r' + ' ' * 50 + '\r')
                    sys.stdout.flush()
                
                return audio_data
                
            return np.array([], dtype=np.float32)
        return np.array([], dtype=np.float32)

def extract_features(audio_data, sample_rate):
    """Extract audio features with improved preprocessing."""
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

        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        features = {}
        
        # Extract MFCCs with more coefficients and delta features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        features['mfcc'] = np.mean(mfccs, axis=1)
        features['delta_mfcc'] = np.mean(delta_mfccs, axis=1)
        features['delta2_mfcc'] = np.mean(delta2_mfccs, axis=1)
        
        # Extract energy and zero crossing rate
        features['energy'] = np.sum(audio_data ** 2) / len(audio_data)
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Add spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
        
        # Store processed audio data
        features['audio_data'] = audio_data
        features['sample_rate'] = sample_rate
        
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