import base64
from io import BytesIO

import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils


class NumpyProcessor:
    def __init__(self):
        self.scale = 1.0 / 255.0
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        x = np.array(images)
        if x.ndim == 3:
            x = x[None, ...]
        if x.ndim != 4 or x.shape[-1] != 3:
            raise RuntimeError()
        if not (x.shape[1] == 256 and x.shape[2] == 256):
            raise ValueError()

        x = x * self.scale
        x = (x - self.mean) / self.std
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
        return x


class TritonPythonModel:
    def __init__(self, *args, **kwargs):
        self.processor = NumpyProcessor()

    def call(self, request):
        b64_image = pb_utils.get_input_tensor_by_name(request, "IMAGE_B64").as_numpy()[
            0
        ]
        jpeg_bytes = base64.b64decode(b64_image)
        img = Image.open(BytesIO(jpeg_bytes))
        img = self.processor(img)
        return img

    def execute(self, requests):
        responses = []
        for request in requests:
            processed_images = self.call(request)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("PREPROCESSED_IMAGE", processed_images),
                ]
            )
            responses.append(inference_response)
        return responses
