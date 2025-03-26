from flask import Flask, Response, request, send_from_directory
from flask_sock import Sock
import struct
import json
import os

# Set environment variables for vLLM
os.environ["VLLM_MAX_MODEL_LEN"] = "100000"
os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.9"
os.environ["VLLM_DISABLE_LOGGING"] = "1"

# Initialize model as None
engine = None

def load_model(model_name="canopylabs/orpheus-tts-0.1-finetune-prod"):
    global engine
    if engine is None:
        try:
            # Import the necessary modules
            from vllm.engine.arg_utils import EngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from orpheus_tts.engine_class import OrpheusModel
            
            # Store the original from_engine_args method
            original_from_engine_args = AsyncLLMEngine.from_engine_args
            
            # Define a patched version that doesn't use disable_log_requests
            def patched_from_engine_args(engine_args, **kwargs):
                # Override the max_model_len in engine_args
                engine_args.max_model_len = 100000
                engine_args.gpu_memory_utilization = 0.9
                
                print(f"Patched from_engine_args called with max_model_len={engine_args.max_model_len}")
                
                # Call the original without any extra kwargs
                return original_from_engine_args(engine_args)
            
            # Replace the class method
            AsyncLLMEngine.from_engine_args = staticmethod(patched_from_engine_args)
            print("Successfully patched AsyncLLMEngine.from_engine_args")
            
            # Initialize the model
            engine = OrpheusModel(model_name=model_name)
            print("Successfully initialized OrpheusModel")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

app = Flask(__name__)
sock = Sock(app)

@app.route('/')
def index():
    return send_from_directory('.', 'client.html')

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@sock.route('/ws')
def websocket_endpoint(ws):
    while True:
        try:
            # Ensure model is loaded
            if engine is None:
                load_model()
                
            data = json.loads(ws.receive())
            prompt = data.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
            
            # Send WAV header first
            ws.send(json.dumps({'type': 'audio_chunk', 'chunk': create_wav_header().hex()}))
            
            # Generate and stream audio chunks
            syn_tokens = engine.generate_speech(
                prompt=prompt,
                voice="tara",
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=2000,
                temperature=0.4,
                top_p=0.9
            )
            
            for chunk in syn_tokens:
                ws.send(json.dumps({'type': 'audio_chunk', 'chunk': chunk.hex()}))
            
            # Send end signal
            ws.send(json.dumps({'type': 'generation_complete'}))
            
        except Exception as e:
            print(f"WebSocket error: {e}")
            break

@app.route('/tts', methods=['GET'])
def tts():
    # Ensure model is loaded
    if engine is None:
        load_model()
        
    prompt = request.args.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')

    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=0.4,
            top_p=0.9
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
