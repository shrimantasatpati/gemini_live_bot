# Real-time Audio Q&A Bot with Gemini Live API

A Streamlit-based application that provides real-time audio question-answering using Google's Gemini 2.5 Flash Live API with custom knowledge base support.

## 🎯 Features

- **Real-time Audio Responses**: Generates and plays audio responses in real-time
- **Multiple Input Methods**: Text, Audio (coming soon), Video (coming soon)
- **Custom Knowledge Base**: Upload your own .md files for domain-specific answers
- **Token Tracking**: Monitor input/output tokens for cost analysis
- **Latency Metrics**: Track first-chunk and total latency
- **Cost Calculator**: Automatic cost calculation per request
- **Conversation History**: View and track all conversations
- **Multiple Output Modes**: Audio only, Text + Audio, Text only

## 📁 Project Structure

```
audio-qa-bot/
├── backend/
│   ├── __init__.py
│   └── audio_processor.py          # Core audio processing logic
├── frontend/
│   └── app.py                       # Streamlit application
├── knowledge_base/
│   └── example_knowledge.md         # Sample knowledge base
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Google Generative AI API key ([Get one here](https://makersuite.google.com/app/apikey))
- PyAudio (requires system audio libraries)

### 1. Install System Dependencies

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows:**
```bash
# PyAudio wheel will be installed via pip
```

### 2. Clone and Setup

```bash
# Create project directory
mkdir audio-qa-bot
cd audio-qa-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file or configure directly in the Streamlit interface:

```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Run the Application

```bash
streamlit run frontend/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### Basic Usage

1. **Enter API Key**: In the sidebar, enter your Google API key
2. **Upload Knowledge Base** (Optional): Upload a .md file with your custom knowledge
3. **Select Output Mode**: Choose between Audio Only, Text + Audio, or Text Only
4. **Ask Questions**: Type your question in the text area
5. **Submit**: Click "Submit" and wait for the response

### Knowledge Base Format

Your knowledge base should be a markdown (.md) file with clear sections:

```markdown
# Company Information

## Services
- Service 1
- Service 2

## Contact
- Phone: XXX-XXX-XXXX
- Email: email@example.com
```

### Understanding Metrics

**Input Tokens**: Number of tokens in your question + knowledge base
**Output Tokens**: Number of tokens in the AI's response
**Total Latency**: Time from request to completion
**First Chunk Latency**: Time to receive first audio chunk (audio mode only)
**Cost**: Calculated based on Gemini 2.5 Flash pricing

## 💰 Pricing Information

Based on Google Gemini 2.5 Flash pricing (as of January 2025):

- **Input**: $0.075 per 1M tokens
- **Output**: $0.30 per 1M tokens

Example costs:
- Short Q&A (100 input / 200 output tokens): ~$0.000068
- Medium conversation (500 input / 1000 output tokens): ~$0.000338
- Long interaction (2000 input / 5000 output tokens): ~$0.0015

## 🔧 Configuration Options

### Audio Parameters

In `backend/audio_processor.py`:
```python
CHANNELS = 1          # Mono audio
SAMPLE_RATE = 24000   # 24kHz (required by Gemini)
SAMPLE_WIDTH = 2      # 16-bit audio
```

### Model Selection

Change the model in `audio_processor.py`:
```python
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
```

Available models:
- `gemini-2.5-flash-native-audio-preview-12-2025` (Latest)
- `gemini-2.5-flash-native-audio-preview-09-2025`

## 📈 Performance Metrics

Typical performance (varies by question complexity):

| Metric | Audio Mode | Text Mode |
|--------|-----------|-----------|
| First Chunk Latency | 200-500ms | N/A |
| Total Latency | 1-3 seconds | 500-1500ms |
| Input Tokens | 50-500 | 50-500 |
| Output Tokens | 100-1000 | 100-1000 |

## 🐛 Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### API Key Issues

- Ensure your API key is valid
- Check API quotas in Google Cloud Console
- Verify billing is enabled for your project

### Audio Not Playing

- Check system audio settings
- Ensure PyAudio is properly installed
- Try running with administrator privileges
- Verify audio output device is connected

### Knowledge Base Not Loading

- Ensure file is .md format
- Check file encoding (should be UTF-8)
- Verify file is not corrupted
- Check file size (keep under 1MB for best performance)

## 🔐 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Monitor API usage for unusual activity

## 📝 API Approach Explanation

This application uses the **Gemini Live API (Multimodal Live API)** approach:

### Why Live API?

1. **Real-time Streaming**: Bidirectional streaming for natural conversations
2. **Native Audio Support**: No need for separate speech-to-text/text-to-speech
3. **Lower Latency**: Audio chunks stream as they're generated
4. **Multimodal Input**: Can handle text, audio, and video simultaneously
5. **Better UX**: Users hear responses as they're being generated

### Live API vs Standard API

| Feature | Live API | Standard API |
|---------|----------|--------------|
| Latency | 200-500ms first chunk | 1-3 seconds |
| Streaming | Yes | No |
| Audio Native | Yes | Requires TTS |
| Real-time | Yes | No |
| Use Case | Conversations | Batch processing |

### Implementation Details

The Live API uses websocket connections:
```python
async with client.aio.live.connect(model=MODEL, config=config) as session:
    # Send message
    await session.send_client_content(...)
    
    # Receive streaming response
    turn = session.receive()
    async for response in turn:
        # Process audio chunks in real-time
        stream.write(response.data)
```

## 🎓 Learning Resources

- [Gemini Live API Documentation](https://ai.google.dev/api/live)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/docs/)

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📧 Support

For issues related to:
- **Gemini API**: Contact Google Cloud Support
- **Application**: Open an issue in this repository


https://github.com/google-gemini/live-api-web-console/issues/142
https://github.com/googleapis/js-genai/issues/1199
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py
https://beta.dashboard.getstream.io/organization/1343474/1480711/chat/?range=30d&from=1768501362845&to=1768501584778
https://github.com/GetStream/Vision-Agents/blob/main/examples/03_phone_and_rag_example/outbound_phone_example.py
https://gist.github.com/amosgyamfi/23197f96c7649909682877733b53395e
https://google.github.io/adk-docs/streaming/#
https://ai.google.dev/gemini-api/docs/live?example=mic-stream