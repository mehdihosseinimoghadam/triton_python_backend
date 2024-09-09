import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "audio_processing"

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
        httpclient.InferRequestedOutput("SPECTROGRAM_OUTPUT"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    spectrogram_bytes = response.as_numpy("SPECTROGRAM_OUTPUT")[0]

    # Save the spectrogram image
    with open("output_spectrogram.png", "wb") as f:
        f.write(spectrogram_bytes)

    print("Mel spectrogram saved as output_spectrogram.png")
    print("PASS: Audio processing")
    sys.exit(0)
