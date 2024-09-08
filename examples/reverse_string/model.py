import json
import numpy as np
import triton_python_backend_utils as pb_utils


class ReverseStringModel:
    def forward(self, input_string):
        return input_string[::-1]


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self.model = ReverseStringModel()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_string = input_tensor.as_numpy()[0].decode("utf-8")

            output_string = self.model.forward(input_string)

            output_tensor = pb_utils.Tensor(
                "OUTPUT",
                np.array([output_string.encode("utf-8")], dtype=self.output_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
