import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizer, BertModel
import torch


class BertEmbeddingModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()

    def get_embedding(self, input_string):
        inputs = self.tokenizer(
            input_string,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = BertEmbeddingModel()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_string = input_tensor.as_numpy()[0].decode("utf-8")

            embedding = self.model.get_embedding(input_string)

            embedding_tensor = pb_utils.Tensor(
                "EMBEDDING", embedding.astype(np.float32)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[embedding_tensor]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
