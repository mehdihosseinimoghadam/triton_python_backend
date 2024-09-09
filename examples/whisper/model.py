import json
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper
import io
import soundfile as sf
import librosa


class AudioTranscriptionModel:
    def __init__(self):
        self.model = whisper.load_model("tiny.en")

    def transcribe_audio(self, audio_data):
        # Load audio from byte data
        with io.BytesIO(audio_data) as audio_file:
            audio, sample_rate = sf.read(audio_file)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Transcribe the audio
        result = self.model.transcribe(audio)
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
