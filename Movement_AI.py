import cv2 
import base64 
import requests 
import smbus 
import os 
import json 
import grpc 
import pyaudio 
import numpy as np 
import itertools 
import threading 
import cv2 
import requests 
import base64 
import re 
import json 
import time 
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc 
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc 
import riva.client.proto.riva_audio_pb2 as audio_pb2 
MEMORY_FILE = "jack_memory.json" 
# Configuration 
SERVER = "grpc.nvcf.nvidia.com:443" 
USE_SSL = True 
AUTH_TOKEN = "Bearer nvapi
KaFuCVgyyFuKeWQ_PD1D8v3XIO3a48Dz6jD0HhgKd0s_DWrUnnnuMqwC
 NSEMIxW-" 
FUNCTION_ID_ASR = "GENERATE FROM NVIDIA RIVA" 
FUNCTION_ID_TTS = "GENERATE FROM NVIDIA RIVA" 
LANGUAGE_CODE = "en-US" 
SAMPLE_RATE = 16000 
CHUNK_SIZE = 4096 
WAKE_WORD = "jack" 
VOICE_NAME = "English-US.Male-1" 
# gRPC setup 
creds = grpc.ssl_channel_credentials() if USE_SSL else None 
channel = grpc.secure_channel(SERVER, creds) if USE_SSL else 
grpc.insecure_channel(SERVER) 
asr_client = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel) 
tts_client = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel) 
# === PCA9685 Servo Config === 
PCA9685_ADDRESS = 0x40 
bus = smbus.SMBus(1) 
RIGHT_CHANNEL = 0 
LEFT_CHANNEL = 1 
UPDOWN_CHANNEL = 2 
LEFTRIGHT_CHANNEL = 3 
RIGHT_STOP = 327 
RIGHT_FORWARD = 302 
RIGHT_BACKWARD = 352 
LEFT_STOP = 325 
LEFT_FORWARD = 525 
LEFT_BACKWARD = 125 
UP_MIN = 10 
DOWN_MAX = 180 
LEFT_MAX = 180 
RIGHT_MIN = 0 
CENTER = 90 
def set_servo(channel, pulse): 
bus.write_byte_data(PCA9685_ADDRESS, 0x06 + 4*channel, 0) 
bus.write_byte_data(PCA9685_ADDRESS, 0x07 + 4*channel, 0) 
bus.write_byte_data(PCA9685_ADDRESS, 0x08 + 4*channel, pulse & 
0xFF) 
29 
  
 
30 
 
    bus.write_byte_data(PCA9685_ADDRESS, 0x09 + 4*channel, pulse >> 8) 
 
# ========== CONFIGURE THIS ========== 
GEMINI_API_URL = 
"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0
flash:generateContent?key="GENERATE FROM NVIDIA RIVA" 
GEMINI_VISION_URL = GEMINI_API_URL 
# ==================================== 
 
interrupt_flag = threading.Event() 
 
 
def audio_stream(): 
    """Stream microphone audio to Riva ASR.""" 
    audio = pyaudio.PyAudio() 
    stream = audio.open(format=pyaudio.paInt16, channels=1, 
rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE) 
    try: 
        while True: 
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False) 
            yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=data) 
    finally: 
        stream.stop_stream() 
        stream.close() 
        audio.terminate() 
 
 
def detect_wake_word(): 
    """Listens for the wake word 'jack' anywhere in the speech and processes 
input accordingly.""" 
    metadata = (('function-id', FUNCTION_ID_ASR), ('authorization', 
AUTH_TOKEN)) 
    config = riva_asr_pb2.RecognitionConfig( 
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM, 
        sample_rate_hertz=SAMPLE_RATE, 
        language_code=LANGUAGE_CODE, 
        max_alternatives=1, 
        enable_automatic_punctuation=False 
    ) 
    streaming_config = 
riva_asr_pb2.StreamingRecognitionConfig(config=config) 
     
    request_iterator = itertools.chain( 
        
iter([riva_asr_pb2.StreamingRecognizeRequest(streaming_config=streaming_c
 onfig)]), 
        audio_stream() 
    ) 
 
    print(f" Listening for wake word: '{WAKE_WORD}'...") 
 
    for response in asr_client.StreamingRecognize(request_iterator, 
metadata=metadata): 
        if response.results: 
            prompt = response.results[0].alternatives[0].transcript.lower().strip() 
            print(f"Detected: {prompt}") 
 
            # Check if "jack" appears anywhere in the sentence 
            words = prompt.split() 
            if WAKE_WORD in words: 
                # Remove "jack" and keep the rest of the sentence 
                remaining_words = [word for word in words if word != 
WAKE_WORD] 
                remaining_text = " ".join(remaining_words).strip() 
 
                if remaining_text: 
                    print(f" Wake word detected inside: '{prompt}'") 
                    interrupt_flag.set()  # Interrupt current speech 
                    transcribe_live(remaining_text)  # Process input excluding 'jack' 
                else: 
                    print(" Only wake word detected. Waiting for full input...") 
                    prompt = transcribe_speech()  # Wait for full input 
                    transcribe_live(prompt) 
 
                return True  # Stop listening after detecting "jack" 
    return False 
 
 
def generate_speech(text): 
    """Convert text to speech and allow interruption by wake word.""" 
    if not text.strip(): 
        return 
 
    global interrupt_flag 
    interrupt_flag.clear() 
 
    metadata = (("function-id", FUNCTION_ID_TTS), ("authorization", 
AUTH_TOKEN)) 
    request = riva_tts_pb2.SynthesizeSpeechRequest( 
        text=re.sub(r'[*#]', '', text), 
        language_code=LANGUAGE_CODE, 
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM, 
        sample_rate_hz=22050, 
        voice_name=VOICE_NAME 
    ) 
    response = tts_client.Synthesize(request, metadata=metadata) 
    if not response.audio: 
        return 
    p = pyaudio.PyAudio() 
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, 
output=True) 
    audio_data = np.frombuffer(response.audio, dtype=np.int16) 
 
    # Start wake word detection in parallel 
    wake_word_thread = threading.Thread(target=detect_wake_word) 
    wake_word_thread.start() 
 
    for i in range(0, len(audio_data), 1024): 
        if interrupt_flag.is_set(): 
            stream.stop_stream() 
            stream.close() 
            p.terminate() 
            print(" Wake word detected! Interrupting speech...") 
 
            wake_word_thread.join() 
            return 
 
        stream.write(audio_data[i:i+1024].tobytes()) 
 
    stream.stop_stream() 
    stream.close() 
    p.terminate() 
    wake_word_thread.join() 
     
 
def transcribe_speech(): 
    """Converts speech to text using Riva ASR.""" 
    metadata = (('function-id', FUNCTION_ID_ASR), ('authorization', 
AUTH_TOKEN)) 
    config = riva_asr_pb2.RecognitionConfig( 
        encoding=audio_pb2.AudioEncoding.LINEAR_PCM, 
  
 
34 
 
        sample_rate_hertz=SAMPLE_RATE, 
        language_code=LANGUAGE_CODE, 
        max_alternatives=1, 
        enable_automatic_punctuation=True 
    ) 
    streaming_config = 
riva_asr_pb2.StreamingRecognitionConfig(config=config) 
     
    request_iterator = itertools.chain( 
        
iter([riva_asr_pb2.StreamingRecognizeRequest(streaming_config=streaming_c
 onfig)]), 
        audio_stream()  # Ensure this is defined earlier 
    ) 
 
    for response in asr_client.StreamingRecognize(request_iterator, 
metadata=metadata): 
        if response.results: 
            transcript = response.results[0].alternatives[0].transcript 
            return transcript  # ✅ Ensure it's returning a string, not a tuple 
    return ""  
 

def capture_image(filename="frame.jpg"): 
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read() 
    cap.release() 
    if not ret: 
        raise RuntimeError("❌ Camera failed") 
  
    cv2.imwrite(filename, frame) 
    return filename 
 
def encode_image_base64(filename): 
    with open(filename, "rb") as f: 
        return base64.b64encode(f.read()).decode("utf-8") 
 
 
def load_memory(): 
    if os.path.exists(MEMORY_FILE): 
        with open(MEMORY_FILE, "r") as f: 
            return json.load(f) 
    return [] 
 
def save_memory(memory): 
    with open(MEMORY_FILE, "w") as f: 
        json.dump(memory, f, indent=2) 
 
conversation_history = load_memory() 
 
def extract_object_name(prompt): 
    payload = { 
        "contents": [{ 
            "role": "user", 
            "parts": [ 
                {"text": ( 
                    "You are Jack, a smart AI created by CORE-X, you have wheels 
and neck for movement Your job is to extract the object to be followed " 
                    "from a human instruction. Only return the object name in 
lowercase — no extra text.\n\n" 
                    f"Instruction: {prompt}" 
                )} 
            ] 
  
        }] 
    } 
    headers = {"Content-Type": "application/json"} 
    response = requests.post(GEMINI_API_URL, headers=headers, 
json=payload) 
    if response.status_code == 200: 
        return 
response.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower() 
    else: 
        return None 
def track_object(object_name): 
    print(f" Looking for: {object_name}") 
     
    # Define color ranges (you can expand this list) 
    COLORS = { 
        "red": ((0, 100, 100), (10, 255, 255)), 
        "blue": ((100, 150, 0), (140, 255, 255)), 
        "green": ((40, 70, 70), (80, 255, 255)), 
        "yellow": ((20, 100, 100), (30, 255, 255)), 
        "black": ((0, 0, 0), (180, 255, 30)), 
        "white": ((0, 0, 200), (180, 25, 255)), 
    } 
 
    found = False 
    neck_angle = CENTER 
    set_servo(LEFTRIGHT_CHANNEL, CENTER) 
     
    cap = cv2.VideoCapture(0) 
 
    # Check for known color 
    for color, (lower, upper) in COLORS.items(): 
        if color in object_name: 
            print(f" Tracking by color: {color}") 
            lower_hsv = cv2.cvtColor(np.uint8([[lower]]), 
cv2.COLOR_BGR2HSV)[0][0] 
            upper_hsv = cv2.cvtColor(np.uint8([[upper]]), 
cv2.COLOR_BGR2HSV)[0][0] 
            break 
    else: 
        print(" Unknown color. Please use a known color in your object 
description.") 
        return 
 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, lower, upper) 
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, 
cv2.CHAIN_APPROX_SIMPLE) 
 
        if contours: 
            largest = max(contours, key=cv2.contourArea) 
            x, y, w, h = cv2.boundingRect(largest) 
            center_x = x + w // 2 
 
            frame_center = frame.shape[1] // 2 
            diff = center_x - frame_center 
 
            if abs(diff) < 30: 
                print("✅ Object centered. Following.") 
                set_servo(LEFT_CHANNEL, LEFT_FORWARD) 
                set_servo(RIGHT_CHANNEL, RIGHT_FORWARD) 
                time.sleep(1) 
                set_servo(LEFT_CHANNEL, LEFT_STOP) 
                set_servo(RIGHT_CHANNEL, RIGHT_STOP) 
                found = True 
                break 
            elif diff > 0 and neck_angle < LEFT_MAX: 
                neck_angle += 10 
                set_servo(LEFTRIGHT_CHANNEL, neck_angle) 
                time.sleep(0.3) 
            elif diff < 0 and neck_angle > RIGHT_MIN: 
                neck_angle -= 10 
                set_servo(LEFTRIGHT_CHANNEL, neck_angle) 
                time.sleep(0.3) 
            else: 
                # Rotate wheels to continue search 
                print(" Rotating wheels to continue scanning...") 
                set_servo(LEFT_CHANNEL, LEFT_FORWARD) 
                set_servo(RIGHT_CHANNEL, RIGHT_BACKWARD) 
                time.sleep(0.8) 
                set_servo(LEFT_CHANNEL, LEFT_STOP) 
                set_servo(RIGHT_CHANNEL, RIGHT_STOP) 
                neck_angle = CENTER 
                set_servo(LEFTRIGHT_CHANNEL, CENTER) 
        else: 
            print(" Scanning...") 
 
    cap.release() 
 
 
def decide_if_vision_needed(prompt): 
    payload = { 
        "contents": [{ 
            "role": "user", 
            "parts": [{ 

                "text": ( 
                    "You are Jack, an AI assistant with a live camera. Reply only 'yes' 
if the prompt needs camera vision " 
                    "(like who’s here, what someone is doing, object location, etc). 
Otherwise, reply only 'no'.\n\n" 
                    f"Prompt: {prompt}" 
                ) 
            }] 
        }] 
    } 
    headers = {"Content-Type": "application/json"} 
    res = requests.post(GEMINI_API_URL, headers=headers, json=payload) 
    if res.status_code == 200: 
        text = 
res.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower() 
        return "yes" in text 
    else: 
        print(f" Vision check failed: {res.status_code}") 
        return False 
 
def ask_jack(prompt, use_vision=False): 
    global conversation_history 
    headers = {"Content-Type": "application/json"} 
 
    conversation_history.append({"role": "user", "text": prompt}) 
    if len(conversation_history) > 10: 
        conversation_history = conversation_history[-10:] 
 
    if use_vision: 
        filename = capture_image() 
        image_b64 = encode_image_base64(filename) 
 
        payload = { 

 
            "contents": [ 
                *[ 
                    {"role": item["role"], "parts": [{"text": item["text"]}]} 
                    for item in conversation_history 
                ], 
                { 
                    "role": "user", 
                    "parts": [ 
                        { 
                            "text": ( 
                               "You are Jack, created by CORE-X. You can see the live 
camera feed. " 
                               "Answer casually in 2–3 short lines. Keep it friendly." 
                            ) 
                        }, 
                        { 
                            "inlineData": { 
                                "mimeType": "image/jpeg", 
                                "data": image_b64 
                            } 
                        }, 
                        {"text": prompt} 
                    ] 
                } 
            ] 
        } 
        response = requests.post(GEMINI_VISION_URL, headers=headers, 
json=payload) 
 
    else: 
        payload = { 
            "contents": [ 
                *[ 
  

                    {"role": item["role"], "parts": [{"text": item["text"]}]} 
                    for item in conversation_history 
                ], 
                { 
                    "role": "user", 
                    "parts": [{ 
                        "text": ( 
                            "You are Jack, created by CORE-X. Reply in 2–3 casual lines 
only. Be short and helpful.\n\n" 
                            f"{prompt}" 
                        ) 
                    }] 
                } 
            ] 
        } 
        response = requests.post(GEMINI_API_URL, headers=headers, 
json=payload) 
 
    if response.status_code == 200: 
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"] 
        conversation_history.append({"role": "assistant", "text": reply}) 
        save_memory(conversation_history) 
        return reply 
    else: 
        return f"❌ Error: {response.status_code}" 
 
 
def transcribe_live(initial_prompt=None): 
    try: 
        while True: 
            if initial_prompt: 
                prompt = initial_prompt 
                initial_prompt = None  # Reset after first use 

            else: 
                prompt = transcribe_speech() 
 
            print(f" You said: {prompt}") 
            if not prompt.strip(): 
                continue 
 
            print(" Deciding if camera input is needed...") 
            use_cam = decide_if_vision_needed(prompt) 
 
            print(f" Camera will {'be used' if use_cam else 'NOT be used'} for 
this prompt.") 
            reply = ask_jack(prompt, use_cam) 
 
            print(f"\n Jack says: {reply}") 
            generate_speech(reply) 
 
            if "follow" in prompt.lower() or "move to" in prompt.lower(): 
                object_name = extract_object_name(prompt) 
                if object_name: 
                    track_object(object_name) 
                else: 
                    print("❌ Couldn't identify object from prompt.") 
 
    except KeyboardInterrupt: 
        print("\n Jack is signing off.") 
 
 
 
if _name_ == "_main_":
    transcribe_live() 
