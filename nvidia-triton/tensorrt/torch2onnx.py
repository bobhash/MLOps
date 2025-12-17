import argparse

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class ModelWrapper(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, images):
        embedding = self.vision_model(
            pixel_values=images,
            output_attentions=False,
            output_hidden_states=False,
            interpolate_pos_encoding=False,
        ).pooler_output
        return embedding.to(torch.float16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help="The path to Transformers SigLip2 model"
    )
    parser.add_argument(
        "--onnx_savepath", required=True, help="The path to save onnx-serialized model"
    )
    parser.add_argument(
        "--sample_image_path",
        required=False,
        default=(
            "/home/toomuch/new-zelda-image-encoder/calib/laion-coco-aesthetic/downloaded"
            "/2d8a8500.jpg"
        ),
    )
    args = parser.parse_args()
    return args.model_path, args.onnx_savepath, args.sample_image_path


def main():
    model_path, onnx_savepath, sample_image_path = parse_args()

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="cpu").eval()
    wrapped = ModelWrapper(model.vision_model)

    image = Image.open(sample_image_path).convert("RGB")
    inputs = processor(images=[image, image], return_tensors="pt")

    torch.onnx.export(
        model=wrapped,
        args=(inputs["pixel_values"].to(torch.float32)),
        f=onnx_savepath,
        verbose=False,
        export_params=True,
        input_names=[
            "IMAGES",
        ],
        output_names=["EMBEDDINGS"],
        opset_version=21,
        dynamic_shapes={
            "images": {
                0: "batch_size",
            },
        },
        do_constant_folding=False,
        dynamo=True,
        external_data=False,
        optimize=True,
        profile=True,
    )


if __name__ == "__main__":
    main()
