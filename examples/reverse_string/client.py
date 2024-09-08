import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "pytorch"

with httpclient.InferenceServerClient("localhost:8000") as client:
    input_string = "Hello, Triton!"
    input_data = np.array([input_string.encode("utf-8")], dtype=object)

    inputs = [
        httpclient.InferInput(
            "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]

    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output_data = response.as_numpy("OUTPUT")[0].decode("utf-8")

    print(f"Input: {input_string}")
    print(f"Reversed: {output_data}")

    if input_string[::-1] != output_data:
        print("pytorch example error: incorrect reversal")
        sys.exit(1)

    print("PASS: pytorch")
    sys.exit(0)
