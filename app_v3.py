import streamlit as st
import asyncio
import os
import io
import wave
import glob
import traceback
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import time

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
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
if "audio_input_key" not in st.session_state:
    st.session_state.audio_input_key = 0  # Counter to reset the widget

def load_instruction_base():
    # If file doesn't exist, create it with the template
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

# --- CORE INTERACTION LOGIC ---
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
    api_start_time = time.time() # Start measuring API time
    input_tokens = 0
    
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            # 1. Send Logic
            if input_type == "text":
                input_tokens = len(input_data) // 4
                await session.send(input=input_data, end_of_turn=True)
                
            elif input_type == "audio":
                # Extract PCM from the uploaded WAV file
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

    # 3. Process Metrics
    # Time to First Token (TTFT) - API Latency
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
st.title("🎙️ Gemini Live Bot")

# 1. Sidebar
with st.sidebar:
    st.header("Context")
    kbs = load_knowledge_bases()
    selected_kb = st.selectbox("Active Knowledge Base", list(kbs.keys()))
    
    # Load raw instruction template
    raw_instruction = load_instruction_base()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.audio_input_key = 0 # Reset key
        st.rerun()

# Build Context with Dynamic Replacement
kb_text = kbs.get(selected_kb, "")

# Replace {company_name} placeholder with the actual selected KB name
formatted_instruction = raw_instruction.replace("{company_name}", selected_kb).replace("{Knowledge base name}", selected_kb)

full_system_instruction = f"{formatted_instruction}\n\nCONTEXT:\n{kb_text}"

# 2. Chat History (Natural Flow)
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
                
                # UPDATE: Use weighted ratios to give 'Cost' more space
                # [TTFT, Total Latency, In Tok, Out Tok, Cost]
                # We give the last column (Cost) a weight of 2 to prevent truncation
                cols = st.columns([1.2, 1.2, 0.8, 0.8, 2])
                
                cols[0].metric("TTFT (API)", f"{m['ttft_latency']:.2f}s", help="Time To First Token from API")
                cols[1].metric("Total Latency", f"{m['total_latency']:.2f}s", help="Round trip time: Input Start -> Output Ready")
                cols[2].metric("In Tok", m["input_tokens"])
                cols[3].metric("Out Tok", m["output_tokens"])
                cols[4].metric("Cost", f"${cost:.5f}")

# 3. Input Controls
st.divider()
tab_text, tab_audio = st.tabs(["⌨️ Text Input", "🎙️ Audio Input"])

# --- TEXT TAB ---
with tab_text:
    text_input = st.chat_input("Type your question here...")
    if text_input:
        st.session_state.chat_history.append({"role": "user", "type": "text", "content": text_input})
        
        # Start Timer for Total Output Latency
        ui_start_time = time.time()
        
        with st.spinner("Gemini is thinking..."):
            text_resp, audio_resp, metrics = asyncio.run(
                generate_response(text_input, "text", full_system_instruction)
            )
            
            # Calculate Total Output Latency
            ui_end_time = time.time()
            metrics["total_latency"] = ui_end_time - ui_start_time

            if metrics.get("error"):
                st.error(f"Error: {metrics['error']}")
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": text_resp,
                    "audio": audio_resp,
                    "metrics": metrics
                })
                st.rerun()

# --- AUDIO TAB ---
with tab_audio:
    # Use dynamic key to allow resetting the widget
    audio_input = st.audio_input(
        "Record your voice", 
        key=f"audio_in_{st.session_state.audio_input_key}"
    )
    
    if audio_input:
        audio_bytes = audio_input.read()
        
        st.session_state.chat_history.append({"role": "user", "type": "audio", "content": audio_bytes})
        
        # Start Timer for Total Output Latency
        ui_start_time = time.time()

        with st.spinner("Streaming PCM data to Gemini..."):
            text_resp, audio_resp, metrics = asyncio.run(
                generate_response(audio_bytes, "audio", full_system_instruction)
            )
            
            # Calculate Total Output Latency
            ui_end_time = time.time()
            metrics["total_latency"] = ui_end_time - ui_start_time
            
            if metrics.get("error"):
                st.error(f"Error: {metrics['error']}")
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": text_resp,
                    "audio": audio_resp,
                    "metrics": metrics
                })
                
                # Increment key to reset the audio widget on rerun
                st.session_state.audio_input_key += 1
                st.rerun()