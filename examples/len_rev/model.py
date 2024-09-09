import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TextProcessingModel:
    def process(self, input_string):
        length = len(input_string)
        reversed_string = input_string[::-1]
        return length, reversed_string


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = TextProcessingModel()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_string = input_tensor.as_numpy()[0].decode("utf-8")

            length, reversed_string = self.model.process(input_string)

            length_tensor = pb_utils.Tensor(
                "LENGTH", np.array([length], dtype=np.int32)
            )
            reversed_tensor = pb_utils.Tensor(
                "REVERSED", np.array([reversed_string.encode("utf-8")], dtype=object)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[length_tensor, reversed_tensor]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
