import json
import numpy as np
import triton_python_backend_utils as pb_utils
import librosa
import matplotlib.pyplot as plt
import io


class AudioProcessingModel:
    def __init__(self):
        self.sr = 22050  # Sample rate
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Number of samples between successive frames
        self.n_mels = 128  # Number of Mel bands

    def process_audio(self, audio_data):
        # Load audio from byte data
        y, _ = librosa.load(io.BytesIO(audio_data), sr=self.sr)

        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            S_dB,
            x_axis="time",
            y_axis="mel",
            sr=self.sr,
            fmax=8000,
            hop_length=self.hop_length,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram")
        plt.tight_layout()

        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_bytes = buf.getvalue()
        plt.close()

        return img_bytes


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = AudioProcessingModel()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_INPUT")
            audio_data = input_tensor.as_numpy()[0]

            img_bytes = self.model.process_audio(audio_data)

            output_tensor = pb_utils.Tensor(
                "SPECTROGRAM_OUTPUT", np.array([img_bytes], dtype=np.object_)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
