import riva.client 
import time 
import sounddevice as sd 
import numpy as np 
import wave 
 
# Initialize Riva Speech Client 
riva_server = "localhost:50051"  # Modify if using a different server 
riva_asr = riva.client.ASRService(riva_server) 
riva_tts = riva.client.TTSService(riva_server) 
 
# Audio Configuration 
SAMPLE_RATE = 16000  # Standard ASR sample rate 
CHANNELS = 1 
DURATION = 5  # Recording duration in seconds 
 
def record_audio(filename): 
    """Records audio for a specified duration and saves it as a .wav file.""" 
    print("Listening...") 

    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAM
PLE_RATE, channels=CHANNELS, dtype=np.int16) 
    sd.wait()  # Wait until recording is finished 
     
    with wave.open(filename, 'wb') as wf: 
        wf.setnchannels(CHANNELS) 
        wf.setsampwidth(2)  # 16-bit audio 
        wf.setframerate(SAMPLE_RATE) 
        wf.writeframes(audio_data.tobytes()) 
 
    print("Audio Recorded.") 
 
def transcribe_audio(filename): 
    """Converts speech to text using NVIDIA Riva ASR.""" 
    with open(filename, "rb") as audio_file: 
        audio_data = audio_file.read() 
 
    response = riva_asr.recognize(audio_data, language_code="en-US") 
    if response.results: 
        text = response.results[0].alternatives[0].transcript 
        print(f"User: {text}") 
        return text 
    return "" 
 
def generate_response(input_text): 
    """Processes AI response using a basic rule-based logic or LLM API call.""" 
    if "hello" in input_text.lower(): 
        return "Hello! How can I assist you today?" 
    elif "who are you" in input_text.lower(): 
        return "I am an AI-powered robotic assistant with depth vision and 3D per
ception." 
    elif "what can you do" in input_text.lower(): 
  
        return "I can recognize objects, navigate autonomously, and respond to 
voice commands." 
    else: 
        return "I'm not sure how to answer that, but I'm learning!" 
 
def synthesize_speech(response_text): 
    """Converts AI response text to speech using NVIDIA Riva TTS.""" 
    audio = riva_tts.synthesize(response_text, language_code="en-US") 
    with open("response.wav", "wb") as f: 
        f.write(audio.audio) 
     
    print(f"AI: {response_text}") 
    sd.play(np.frombuffer(audio.audio, dtype=np.int16), samplerate=SAM
PLE_RATE) 
    sd.wait() 
 
# Main Loop 
while True: 
    record_audio("input.wav")  # Capture user input 
    text = transcribe_audio("input.wav")  # Convert speech to text 
    if text.lower() in ["exit", "quit", "stop"]: 
        print("Exiting AI system.") 
        break 
    response = generate_response(text)  # Generate AI response 
    synthesize_speech(response)  # Convert text to speech
