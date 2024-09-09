import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "len_rev"

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
        httpclient.InferRequestedOutput("LENGTH"),
        httpclient.InferRequestedOutput("REVERSED"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    length = response.as_numpy("LENGTH")[0]
    reversed_string = response.as_numpy("REVERSED")[0].decode("utf-8")

    print(f"Input: {input_string}")
    print(f"Length: {length}")
    print(f"Reversed: {reversed_string}")

    if len(input_string) != length or input_string[::-1] != reversed_string:
        print("Text processing example error: incorrect results")
        sys.exit(1)

    print("PASS: Text processing")
    sys.exit(0)
