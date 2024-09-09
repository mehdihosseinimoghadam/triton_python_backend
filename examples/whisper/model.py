import json
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper
import io


class AudioTranscriptionModel:
    def __init__(self):
        self.model = whisper.load_model("tiny.en")

    def transcribe_audio(self, audio_data):
        # Load audio from byte data
        with io.BytesIO(audio_data) as audio_file:
            # Transcribe the audio
            result = self.model.transcribe(audio_file)
        return result["text"]


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = AudioTranscriptionModel()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_INPUT")
            audio_data = input_tensor.as_numpy()[0]

            transcription = self.model.transcribe_audio(audio_data)

            output_tensor = pb_utils.Tensor(
                "TRANSCRIPTION_OUTPUT",
                np.array([transcription.encode("utf-8")], dtype=np.object_),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
