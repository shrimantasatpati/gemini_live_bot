#gemini_live.py
import asyncio
import wave
import contextlib
import pyaudio
from google import genai
from google.genai import types

api_key = ''

# Audio parameters from the Live API
CHANNELS = 1
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit audio

# Wave file writer helper (for file saving)
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

async def async_enumerate(aiterable):
    """Helper to enumerate async iterables"""
    n = 0
    async for item in aiterable:
        yield n, item
        n += 1

async def play_audio_realtime(prompt_text, save_to_file=False, output_filename="output.wav"):
    """
    Generates and plays audio in REAL-TIME using PyAudio.
    Audio plays as chunks arrive - no waiting!
    
    Args:
        prompt_text (str): The text to convert to audio
        save_to_file (bool): Also save to WAV file?
        output_filename (str): The output WAV file path (if saving)
    """
    try:
        client = genai.Client(api_key=api_key)
        # MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'
        # MODEL = 'gemini-live-2.5-flash-native-audio'
        MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
        
        config = {
            "response_modalities": ["AUDIO"]
        }        
        # Initialize PyAudio for real-time playback
        p = pyaudio.PyAudio()
        
        # Open audio stream for playback
        stream = p.open(
            format=p.get_format_from_width(SAMPLE_WIDTH),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True  # output=True means playback
        )        
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            wav_file_context = wave_file(output_filename) if save_to_file else contextlib.nullcontext()            
            with wav_file_context as wav:
                # await session.send_client_content(
                #     turns={"role": "user", "parts": [{"text": prompt_text}]}, 
                #     turn_complete=True
                # )
                await session.send(input=prompt_text, end_of_turn=True)
                
                # Receive and play audio chunks in real-time
                turn = session.receive()
                async for n, response in async_enumerate(turn):
                    if response.data is not None:
                        # PLAY audio immediately (real-time!)
                        stream.write(response.data)
                        
                        # Also save to file if requested
                        if save_to_file and wav:
                            wav.writeframes(response.data)
                        
                        if n == 0:
                            mime_type = response.server_content.model_turn.parts[0].inline_data.mime_type
                            print(f"\n✓ Playing audio in real-time (MIME type: {mime_type})...")
                if save_to_file:
                    print(f"✓ Audio also saved to {output_filename}")
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
                
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

async def save_audio_to_file(prompt_text, output_filename="output.wav"):
    """
    Alternative: Just save to file without real-time playback.
    Use this if you don't need immediate playback.
    """
    try:
        client = genai.Client(api_key=api_key)
        MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'
        
        config = {
            "response_modalities": ["AUDIO"]
        }        
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            with wave_file(output_filename) as wav:
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": prompt_text}]}, 
                    turn_complete=True
                )
                
                turn = session.receive()
                async for n, response in async_enumerate(turn):
                    if response.data is not None:
                        wav.writeframes(response.data)
                
                print(f"\n✓ Audio saved to {output_filename}")
        return True
                
    except Exception as e:
        return False

async def main():    
    prompt = "Hello! This is a real-time audio test. The audio you're hearing is being played as it's generated, with no delay!"
    
    # Option 1: Real-time playback
    print("=" * 60)
    print("OPTION 1: REAL-TIME PLAYBACK")
    print("=" * 60)
    # await play_audio_realtime(prompt, save_to_file=True)
    await play_audio_realtime(prompt, save_to_file=False)
    
    # # Option 2: Just save to file
    # print("=" * 60)
    # print("OPTION 2: SAVE TO FILE ONLY")
    # print("=" * 60)
    # await save_audio_to_file(prompt, "output2.wav")

if __name__ == "__main__":

    asyncio.run(main())
