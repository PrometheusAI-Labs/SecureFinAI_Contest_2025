#!/usr/bin/env python3
"""
Cloud.ru Whisper API Transcriber for FinGPT Compliance Agents
Supports Cloud.ru Foundation Models API for audio transcription
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import io
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CloudRuTranscriber:
    """Audio transcription using Cloud.ru Foundation Models API"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API configuration
        api_keys = self.config.get('api_keys', {})
        self.api_key = api_keys.get('cloud_ru_api_key') or os.getenv('CLOUD_RU_API_KEY') or os.getenv('API_KEY')
        self.base_url = api_keys.get('cloud_ru_base_url') or os.getenv('CLOUD_RU_BASE_URL', 'https://foundation-models.api.cloud.ru/v1')
        
        # Audio processing settings
        self.target_sr = 16000
        
        if not self.api_key:
            logger.warning("Cloud.ru API key not found. Please set CLOUD_RU_API_KEY or API_KEY in your .env file.")
    
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
    
    def transcribe_audio(self, audio_array: np.ndarray, sampling_rate: int, 
                        model: str = "openai/whisper-large-v3", 
                        language: str = "ru",
                        temperature: float = 0.5,
                        response_format: str = "json") -> str:
        """
        Transcribe audio using Cloud.ru Foundation Models API
        
        Args:
            audio_array: Audio data as numpy array
            sampling_rate: Original sampling rate
            model: Model to use for transcription
            language: Language code (ru, en, etc.)
            temperature: Temperature for generation
            response_format: Response format ("text" or "json")
            
        Returns:
            Transcribed text
        """
        if not self.api_key:
            logger.error("Cloud.ru API key not found")
            return ""
        
        try:
            # Convert audio to WAV bytes
            wav_bytes = self.audio_to_wav_bytes(audio_array, sampling_rate)
            
            # Prepare API request
            url = f"{self.base_url}/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Prepare files and data
            files = {
                "file": ("audio.wav", io.BytesIO(wav_bytes), "application/octet-stream")
            }
            
            data = {
                "model": model,
                "response_format": response_format,
                "temperature": str(temperature),
                "language": language
            }
            
            # Make API request
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                if response_format == "text":
                    # For text format, response is plain text
                    transcription = response.text.strip()
                else:
                    # For json format, parse JSON
                    result = response.json()
                    if isinstance(result, dict):
                        transcription = result.get('text', '')
                    else:
                        transcription = str(result)
                
                logger.info(f"Transcription successful: {len(transcription)} characters")
                return transcription.strip()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
    
    def transcribe_batch(self, audio_data: List[Dict], 
                        model: str = "openai/whisper-large-v3",
                        language: str = "ru") -> List[Dict]:
        """
        Transcribe a batch of audio samples
        
        Args:
            audio_data: List of audio samples with 'audio' and 'sampling_rate' keys
            model: Model to use for transcription
            language: Language code
            
        Returns:
            List of results with transcriptions
        """
        results = []
        
        for i, sample in enumerate(audio_data):
            logger.info(f"Transcribing sample {i+1}/{len(audio_data)}")
            
            audio_array = sample['audio']['array']
            sampling_rate = sample['audio']['sampling_rate']
            
            transcription = self.transcribe_audio(audio_array, sampling_rate, model, language)
            
            result = {
                'sample_id': sample.get('sample_id', i),
                'transcription': transcription,
                'success': bool(transcription),
                'original_sample': sample
            }
            
            results.append(result)
        
        return results
    
    def test_connection(self) -> bool:
        """Test connection to Cloud.ru API"""
        if not self.api_key:
            logger.error("No API key configured")
            return False
        
        try:
            # Test with a simple audio sample
            sample_rate = 16000
            duration = 0.1  # Very short sample
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_array = np.sin(2 * np.pi * 440 * t) * 0.1
            
            # Try to transcribe
            transcription = self.transcribe_audio(audio_array, sample_rate)
            
            if transcription is not None:
                logger.info(f"Connection test successful. API responded with: '{transcription}'")
                return True
            else:
                logger.error("Connection test failed: No response from API")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

def main():
    """Test the Cloud.ru transcriber"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Cloud.ru Transcriber')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection')
    parser.add_argument('--model', default='openai/whisper-large-v3', help='Model to use')
    parser.add_argument('--language', default='ru', help='Language code')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize transcriber
    transcriber = CloudRuTranscriber(args.config)
    
    if args.test_connection:
        success = transcriber.test_connection()
        if success:
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed!")
    else:
        print("Cloud.ru Transcriber initialized successfully!")
        print(f"Base URL: {transcriber.base_url}")
        print(f"API Key configured: {'Yes' if transcriber.api_key else 'No'}")
        print(f"Model: {args.model}")
        print(f"Language: {args.language}")

if __name__ == "__main__":
    main()