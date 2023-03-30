from transformers import pipeline
import torch
import whisper_timestamped as whisper
import json
import requests

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)


def callback(projectId, result):
    url = 'https://scribewave.com/api/whisper/banana/callback'
    headers = {'Content-type': 'application/json'}
    data = {
        "projectId": projectId,
        "transcribeResult": result,
        "secret": "curieuze5neuze8mosterdpot33"
    }
    response = requests.post(url, json=data, headers=headers)
    print(response.json())

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    url = model_inputs.get('url', None)
    projectId = model_inputs.get('projectId', None)
    if url == None:
        return {'message': "No url provided"}

    # Run the model
    audio = whisper.load_audio(url)
    result = whisper.transcribe(model, audio)

    # HTTP callback to api to confirm result
    if projectId != None:
        callback(projectId, result)

    # Return the results as a dictionary
    return result

