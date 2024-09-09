import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "bert_embedding"

with httpclient.InferenceServerClient("localhost:8000") as client:
    input_string = "Hello, BERT!"
    input_data = np.array([input_string.encode("utf-8")], dtype=object)

    inputs = [
        httpclient.InferInput(
            "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("EMBEDDING"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    embedding = response.as_numpy("EMBEDDING")

    print(f"Input: {input_string}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values of embedding: {embedding[:5]}")

    if embedding.shape != (768,):
        print("BERT embedding example error: incorrect embedding shape")
        sys.exit(1)

    print("PASS: BERT embedding")
    sys.exit(0)
