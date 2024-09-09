import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "whisper"

with httpclient.InferenceServerClient("localhost:8000") as client:
    # Read .wav file
    with open("audio.wav", "rb") as audio_file:
        audio_data = audio_file.read()

    input_data = np.array([audio_data], dtype=object)

    inputs = [
        httpclient.InferInput(
            "AUDIO_INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("TRANSCRIPTION_OUTPUT"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    transcription = response.as_numpy("TRANSCRIPTION_OUTPUT")[0].decode("utf-8")

    print("Transcription:")
    print(transcription)
    print("PASS: Audio transcription")
    sys.exit(0)
