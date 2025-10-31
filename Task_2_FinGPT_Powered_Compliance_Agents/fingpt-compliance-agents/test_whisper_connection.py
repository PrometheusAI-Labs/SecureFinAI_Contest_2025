#!/usr/bin/env python3
"""
Test script for custom Whisper API connection
Tests connection to different OpenAI-compatible endpoints
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_collection.cloud_ru_transcriber import CloudRuTranscriber
from data_collection.audio_transcriber import AudioTranscriber

def test_connection(provider: str = "cloud_ru", base_url: str = None, api_key: str = None):
    """Test connection to a specific Whisper API endpoint"""
    
    print(f"🔍 Testing {provider.upper()} API connection...")
    print(f"Base URL: {base_url or 'Default'}")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print("-" * 50)
    
    # Create transcriber based on provider
    if provider == "cloud_ru":
        transcriber = CloudRuTranscriber()
        if base_url:
            transcriber.base_url = base_url
        if api_key:
            transcriber.api_key = api_key
    else:  # openai
        transcriber = AudioTranscriber()
        if base_url:
            transcriber.base_url = base_url
            transcriber.client = transcriber._create_openai_client()
        if api_key:
            transcriber.api_key = api_key
            transcriber.client = transcriber._create_openai_client()
    
    # Test connection
    success = transcriber.test_connection()
    
    if success:
        print("✅ Connection successful!")
        print("🎯 You can use this endpoint for audio transcription")
    else:
        print("❌ Connection failed!")
        print("💡 Check your API key and base URL")
    
    return success

def test_with_sample_audio(provider: str = "cloud_ru"):
    """Test transcription with sample audio data"""
    import numpy as np
    
    print(f"\n🎵 Testing transcription with sample audio using {provider.upper()}...")
    
    # Create transcriber based on provider
    if provider == "cloud_ru":
        transcriber = CloudRuTranscriber()
    else:
        transcriber = AudioTranscriber()
    
    if not transcriber.api_key:
        print("❌ No API key available")
        return False
    
    # Create sample audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = np.sin(2 * np.pi * frequency * t) * 0.1
    
    # Test transcription
    try:
        transcription = transcriber.transcribe_audio(audio_array, sample_rate)
        print(f"📝 Transcription result: '{transcription}'")
        
        if transcription:
            print("✅ Audio transcription working!")
            return True
        else:
            print("⚠️ No transcription returned")
            return False
            
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Whisper API Connection')
    parser.add_argument('--provider', choices=['cloud_ru', 'openai'], default='cloud_ru', 
                       help='API provider to test (default: cloud_ru)')
    parser.add_argument('--base-url', help='Custom base URL for Whisper API')
    parser.add_argument('--api-key', help='API key (overrides .env file)')
    parser.add_argument('--test-audio', action='store_true', help='Test with sample audio')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    print("🎯 FinGPT Compliance Agents - Whisper API Tester")
    print("=" * 60)
    
    # Test connection
    connection_success = test_connection(args.provider, args.base_url, args.api_key)
    
    # Test audio transcription if requested
    if args.test_audio and connection_success:
        audio_success = test_with_sample_audio(args.provider)
    else:
        audio_success = False
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"  Provider: {args.provider.upper()}")
    print(f"  Connection: {'✅ Success' if connection_success else '❌ Failed'}")
    print(f"  Audio Test: {'✅ Success' if audio_success else '❌ Skipped/Failed'}")
    
    if connection_success:
        print(f"\n🚀 Your {args.provider.upper()} API is ready to use!")
        print("   You can now run: make test-audio")
    else:
        print("\n💡 Troubleshooting:")
        if args.provider == "cloud_ru":
            print("   1. Check your CLOUD_RU_API_KEY in .env file")
            print("   2. Verify the base URL is correct")
            print("   3. Ensure the endpoint supports Cloud.ru Foundation Models API")
        else:
            print("   1. Check your OPENAI_API_KEY in .env file")
            print("   2. Verify the base URL is correct")
            print("   3. Ensure the endpoint supports OpenAI-compatible API")
        print("   4. Check network connectivity")

if __name__ == "__main__":
    main()