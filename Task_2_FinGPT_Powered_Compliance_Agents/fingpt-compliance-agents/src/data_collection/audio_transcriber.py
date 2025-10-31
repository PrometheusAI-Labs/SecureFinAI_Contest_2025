#!/usr/bin/env python3
"""
Audio Transcriber for FinGPT Compliance Agents
Supports both OpenAI API and custom Whisper endpoints
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import io
import numpy as np
import librosa
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """Audio transcription with support for custom OpenAI endpoints"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API configuration
        api_keys = self.config.get('api_keys', {})
        self.api_key = api_keys.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = api_keys.get('openai_base_url') or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # Initialize OpenAI client
        self.client = self._create_openai_client()
        
        # Audio processing settings
        self.target_sr = 16000
        
    def _create_openai_client(self) -> Optional[OpenAI]:
        """Create OpenAI client with custom base_url if specified"""
        if not self.api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            return None
        
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"OpenAI client initialized with base_url: {self.base_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            return None
    
    def audio_to_wav_bytes(self, audio_array: np.ndarray, sampling_rate: int) -> bytes:
        """
        Clean + convert arbitrary audio to mono, 16kHz, 16-bit PCM WAV bytes.
        Ensures: finite samples, mono mixdown, resampled to 16k, PCM_16 WAV.
        """
        # Ensure numpy array
        x = np.asarray(audio_array)
        if x.ndim == 0 or x.size == 0:
            raise ValueError("Audio array is empty")

        # Remove NaN/Inf
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to float32 in [-1, 1]
        if np.issubdtype(x.dtype, np.integer):
            maxv = float(np.iinfo(x.dtype).max)
            if maxv <= 0:
                raise ValueError(f"Invalid integer dtype range for audio: {x.dtype}")
            x = x.astype(np.float32) / maxv
        else:
            x = x.astype(np.float32)

        # Downmix to mono if multi-channel
        if x.ndim > 1:
            # (T, C) or (C, T) -> heuristically average over channel axis
            if x.shape[-1] <= 8 and x.shape[0] > x.shape[-1]:
                # likely (T, C)
                x = np.mean(x, axis=-1)
            else:
                # likely (C, T)
                x = np.mean(x, axis=0)

        # Resample to target_sr if needed
        sr = int(sampling_rate)
        if sr != self.target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Clip to [-1, 1]
        x = np.clip(x, -1.0, 1.0)

        # Write PCM_16 WAV to bytes
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    
    def wav_bytes_to_base64(self, wav_bytes: bytes) -> str:
        """Convert WAV bytes to base64 string"""
        return base64.b64encode(wav_bytes).decode("utf-8")
    
    def transcribe_audio(self, audio_array: np.ndarray, sampling_rate: int, 
                        model: str = "gpt-4o-audio-preview") -> str:
        """
        Transcribe audio using OpenAI API (supports custom endpoints)
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Original sampling rate
            model: Model to use for transcription
            
        Returns:
            Transcribed text
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return ""
        
        try:
            # Convert audio to WAV bytes
            wav_bytes = self.audio_to_wav_bytes(audio_array, sampling_rate)
            audio_b64 = self.wav_bytes_to_base64(wav_bytes)
            
            # Transcribe using OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please transcribe this audio to text. Only provide the transcription without any additional commentary."
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
            )

            # Extract transcription
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                logger.error("No transcription in response")
                return ""

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            if hasattr(e, 'response'):
                logger.error(f"Response details: {e.response}")
            return ""
    
    def transcribe_batch(self, audio_data: List[Dict], model: str = "gpt-4o-audio-preview") -> List[Dict]:
        """
        Transcribe a batch of audio samples
        
        Args:
            audio_data: List of audio samples with 'audio' and 'sampling_rate' keys
            model: Model to use for transcription
            
        Returns:
            List of results with transcriptions
        """
        results = []
        
        for i, sample in enumerate(audio_data):
            logger.info(f"Transcribing sample {i+1}/{len(audio_data)}")
            
            audio_array = sample['audio']['array']
            sampling_rate = sample['audio']['sampling_rate']
            
            transcription = self.transcribe_audio(audio_array, sampling_rate, model)
            
            result = {
                'sample_id': sample.get('sample_id', i),
                'transcription': transcription,
                'success': bool(transcription),
                'original_sample': sample
            }
            
            results.append(result)
        
        return results
    
    def test_connection(self) -> bool:
        """Test connection to the configured API endpoint"""
        if not self.client:
            return False
        
        try:
            # Try to list models to test connection
            models = self.client.models.list()
            logger.info(f"Successfully connected to {self.base_url}")
            logger.info(f"Available models: {[model.id for model in models.data[:5]]}")  # Show first 5 models
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

def main():
    """Test the audio transcriber"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Audio Transcriber')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize transcriber
    transcriber = AudioTranscriber(args.config)
    
    if args.test_connection:
        success = transcriber.test_connection()
        if success:
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed!")
    else:
        print("Audio Transcriber initialized successfully!")
        print(f"Base URL: {transcriber.base_url}")
        print(f"API Key configured: {'Yes' if transcriber.api_key else 'No'}")

if __name__ == "__main__":
    main()