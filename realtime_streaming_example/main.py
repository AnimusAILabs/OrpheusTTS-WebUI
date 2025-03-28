from flask import Flask, Response, request, send_from_directory
from flask_sock import Sock
import struct
import json
import os
import time
import uuid
import numpy as np

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

@sock.route('/ws')
def websocket_endpoint(ws):
    while True:
        try:
            # Ensure model is loaded
            if engine is None:
                load_model()
                
            data = json.loads(ws.receive())
            prompt = data.get('prompt', 'Hey there, looks like you forgot to provide a prompt!')
            
            # Generate a unique context ID for this audio stream
            context_id = str(uuid.uuid4())
            
            # Send start signal with context ID
            ws.send(json.dumps({
                'type': 'start',
                'context_id': context_id
            }))
            
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
            
            # Buffer for accumulating audio data
            audio_buffer = []
            chunk_size = 16384  # Increased chunk size (about 0.68 seconds at 24kHz)
            min_chunk_size = 8192  # Minimum chunk size before sending
            
            for chunk in syn_tokens:
                audio_buffer.extend(chunk)
                
                # Send chunks when we have enough data
                while len(audio_buffer) >= chunk_size:
                    chunk_to_send = audio_buffer[:chunk_size]
                    audio_buffer = audio_buffer[chunk_size:]
                    
                    ws.send(json.dumps({
                        'type': 'audio_chunk',
                        'context_id': context_id,
                        'chunk': bytes(chunk_to_send).hex()
                    }))
            
            # Send any remaining audio data
            if audio_buffer:
                ws.send(json.dumps({
                    'type': 'audio_chunk',
                    'context_id': context_id,
                    'chunk': bytes(audio_buffer).hex()
                }))
            
            # Send end signal
            ws.send(json.dumps({
                'type': 'generation_complete',
                'context_id': context_id
            }))
            
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
