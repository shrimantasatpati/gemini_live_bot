import streamlit as st
import asyncio
import os
import io
import wave
import glob
import traceback
import time
import av
import numpy as np
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

load_dotenv()

# --- CONFIGURATION ---
API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") # Support Streamlit Secrets
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
KNOWLEDGE_BASE_DIR = "knowledge_bases"
INSTRUCTION_FILE = "instruction.md"

# Audio Config
RECEIVE_SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE_SEND = 4096 

# Pricing
COST_PER_1M_INPUT_TEXT = 0.10
COST_PER_1M_OUTPUT_TEXT = 0.40

# --- SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- INSTRUCTION LOADING ---
def load_instruction_base():
    # Updated Template per user request
    default_template = """You are Genie, a warm, approachable, and professional AI assistant representing company {company_name}. 

**Your Role**
Answer potential customer questions about the company’s services using ONLY the knowledge base below. Do not use outside information.

**Tone**
- Warm, approachable, knowledgeable, and positive
- Professional and trustworthy
- Concise but informative

**Goals**
1. Answer customer questions about services, coverage areas, pricing, and contact options strictly from the knowledge base.
2. If the customer expresses interest in booking services, politely collect:
   - Full name
   - Phone number
   - Email address
   - Service address (where work is needed)
3. Once collected, immediately call the schedule_appointment tool with those values.
4. Confirm the information back to the customer.
5. Close with a warm thank-you message, reassuring them that {company_name} looks forward to helping.
6. At the end of every conversation, call the send_call_summary tool with a summary of the discussion, including key details, answers provided, and actions taken.

**Important Rules**
- Do not invent or assume services not listed in the knowledge base.
- Do not provide personal opinions or unrelated information.
- If asked about something not covered, reply: 
  “That specific detail isn’t available with me now, but I’d be happy to pass your question along to the owner when I schedule your appointment.”
- Always keep interactions professional, customer-focused, and trustworthy."""
    
    if not os.path.exists(INSTRUCTION_FILE):
        with open(INSTRUCTION_FILE, "w", encoding="utf-8") as f:
            f.write(default_template)
            
    with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        return f.read()

def load_knowledge_bases():
    kbs = {}
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "default.md"), "w") as f:
            f.write("No specific knowledge base loaded.")
    
    files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as r:
                kbs[Path(f).stem] = r.read()
        except Exception: pass
    if not kbs: kbs["default"] = "No specific knowledge base loaded."
    return kbs

def pcm_to_wav(pcm_data, sample_rate=RECEIVE_SAMPLE_RATE):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2) 
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()

def calculate_cost(input_tok, output_tok):
    return ((input_tok / 1_000_000) * COST_PER_1M_INPUT_TEXT) + \
           ((output_tok / 1_000_000) * COST_PER_1M_OUTPUT_TEXT)

# --- GEMINI INTERACTION ---
async def generate_response(input_data, input_type, system_instruction):
    client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})
    
    config = {
        "response_modalities": ["AUDIO"], 
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}
    }

    cumulative_text = ""
    cumulative_pcm = bytearray()
    first_token_time = None
    api_start_time = time.time()
    input_tokens = 0
    
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            # 1. Send Logic
            if input_type == "text":
                input_tokens = len(input_data) // 4
                await session.send(input=input_data, end_of_turn=True)
                
            elif input_type == "audio":
                # input_data is standard WAV bytes here
                with wave.open(io.BytesIO(input_data), 'rb') as w:
                    sample_rate = w.getframerate()
                    raw_pcm_data = w.readframes(w.getnframes())
                    duration = w.getnframes() / w.getframerate()
                    input_tokens = int(duration * 25)

                mime_type = f"audio/pcm;rate={sample_rate}"
                
                with io.BytesIO(raw_pcm_data) as pcm_stream:
                    while True:
                        chunk = pcm_stream.read(CHUNK_SIZE_SEND)
                        if not chunk:
                            break
                        is_last = (pcm_stream.tell() == len(raw_pcm_data))
                        await session.send(input={"data": chunk, "mime_type": mime_type}, end_of_turn=is_last)

            # 2. Receive Logic
            async for response in session.receive():
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if first_token_time is None and (part.text or part.inline_data):
                            first_token_time = time.time()
                        
                        if part.text:
                            cumulative_text += part.text
                        if part.inline_data:
                            cumulative_pcm.extend(part.inline_data.data)
                    
                    if response.server_content.turn_complete:
                        break
                        
    except Exception as e:
        tb_str = traceback.format_exc()
        return None, None, {"error": f"{str(e)}\n\n{tb_str}"}

    # 3. Metrics
    ttft_latency = (first_token_time - api_start_time) if first_token_time else 0.0
    output_tokens = len(cumulative_text) // 4 if cumulative_text else 0
    
    metrics = {
        "ttft_latency": ttft_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    
    wav_data = pcm_to_wav(bytes(cumulative_pcm)) if cumulative_pcm else None
    return cumulative_text, wav_data, metrics

# --- UI LAYOUT ---
st.set_page_config(page_title="Gemini Audio Chat", layout="centered")
st.title("🎙️ Gemini Live Bot (WebRTC)")

# 1. Sidebar
with st.sidebar:
    st.header("Context")
    kbs = load_knowledge_bases()
    selected_kb = st.selectbox("Active Knowledge Base", list(kbs.keys()))
    
    raw_instruction = load_instruction_base()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Build Context
kb_text = kbs.get(selected_kb, "")
formatted_instruction = raw_instruction.replace("{company_name}", selected_kb).replace("{Knowledge base name}", selected_kb)
full_system_instruction = f"{formatted_instruction}\n\nCONTEXT:\n{kb_text}"

# 2. Chat History
if not st.session_state.chat_history:
    st.info("Start the conversation below using Text or Audio.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if msg["type"] == "text":
                st.write(msg["content"])
            elif msg["type"] == "audio":
                st.audio(msg["content"], format="audio/wav")
        
        elif msg["role"] == "assistant":
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")
            
            if msg.get("metrics"):
                m = msg["metrics"]
                cost = calculate_cost(m["input_tokens"], m["output_tokens"])
                cols = st.columns([1.2, 1.2, 0.8, 0.8, 2])
                cols[0].metric("TTFT", f"{m['ttft_latency']:.2f}s")
                cols[1].metric("Total", f"{m['total_latency']:.2f}s")
                cols[2].metric("In", m["input_tokens"])
                cols[3].metric("Out", m["output_tokens"])
                cols[4].metric("Cost", f"${cost:.5f}")

st.divider()
tab_text, tab_audio = st.tabs(["⌨️ Text Input", "🎙️ WebRTC Audio"])

# --- TEXT TAB ---
with tab_text:
    text_input = st.chat_input("Type your question here...")
    if text_input:
        st.session_state.chat_history.append({"role": "user", "type": "text", "content": text_input})
        ui_start_time = time.time()
        with st.spinner("Gemini is thinking..."):
            text_resp, audio_resp, metrics = asyncio.run(
                generate_response(text_input, "text", full_system_instruction)
            )
            ui_end_time = time.time()
            metrics["total_latency"] = ui_end_time - ui_start_time
            if metrics.get("error"):
                st.error(f"Error: {metrics['error']}")
            else:
                st.session_state.chat_history.append({"role": "assistant", "text": text_resp, "audio": audio_resp, "metrics": metrics})
                st.rerun()

# --- WEBRTC TAB ---
with tab_audio:
    st.write("Click 'Start' to record. Click 'Stop' to send.")
    
    # Simple Audio Processor to hold frames
    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame)
            return frame

    # Setup WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Crucial for Cloud
    )

    # State management for processing recording after stop
    if "webrtc_processing" not in st.session_state:
        st.session_state.webrtc_processing = False

    if webrtc_ctx.state.playing:
        st.session_state.webrtc_processing = True
    
    # Logic: If we were recording (processing=True) and now state.playing is False (Stopped), process it.
    if not webrtc_ctx.state.playing and st.session_state.webrtc_processing:
        st.session_state.webrtc_processing = False # Reset flag
        
        if webrtc_ctx.audio_processor:
            audio_frames = webrtc_ctx.audio_processor.frames
            
            if audio_frames:
                # Convert AV frames to WAV bytes
                output_buffer = io.BytesIO()
                with av.open(output_buffer, 'w', 'wav') as container:
                    stream = container.add_stream('pcm_s16le', rate=48000, layout='mono')
                    for frame in audio_frames:
                        frame.pts = None # Reset timing
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                
                wav_bytes = output_buffer.getvalue()
                
                # Update UI
                st.session_state.chat_history.append({"role": "user", "type": "audio", "content": wav_bytes})
                
                ui_start_time = time.time()
                with st.spinner("Processing WebRTC Audio..."):
                    text_resp, audio_resp, metrics = asyncio.run(
                        generate_response(wav_bytes, "audio", full_system_instruction)
                    )
                    
                    ui_end_time = time.time()
                    metrics["total_latency"] = ui_end_time - ui_start_time
                    
                    if metrics.get("error"):
                        st.error(f"Error: {metrics['error']}")
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "text": text_resp, "audio": audio_resp, "metrics": metrics})
                        st.rerun()
            else:
                st.warning("No audio recorded. Please speak before stopping.")