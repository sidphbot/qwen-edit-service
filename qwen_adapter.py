import os
import io
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from diffusers import (
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
    BitsAndBytesConfig as DBits,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig as TBits,
)


class QwenImageEdit:
    def __init__(
        self,
        backend: str = "local",
        model_id: Optional[str] = None,
        device: str = "cuda",
    ):
        self.backend = backend
        self.model_id = model_id or os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit")
        self.device = device

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        use_4bit = os.environ.get("USE_4BIT", "0") == "1"

        print(f"[QwenImageEdit] Loading model_id={self.model_id}, device={device}, 4bit={use_4bit}")

        quant_args = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

        # transformer
        if use_4bit:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=DBits(**quant_args),
                torch_dtype=torch_dtype,
            )
        else:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )

        # text encoder
        if use_4bit:
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                quantization_config=TBits(**quant_args),
                torch_dtype=torch_dtype,
            )
        else:
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=torch_dtype,
            )

        # pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
        )

        # load LoRA (Lightning)
        try:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
            )
            print("[LoRA] Loaded lightx2v/Qwen-Image-Lightning (8steps V1.1)")
        except Exception as e:
            print("[LoRA] Skipped:", e)

        pipe.enable_model_cpu_offload()
        self.pipe = pipe
        self.generator = torch.Generator(device=device).manual_seed(42)

    # -------------------------
    # Public API
    # -------------------------
    async def edit_async(
        self,
        prompt: str,
        image_bytes: bytes,
        outdir: Path,
        num_inference_steps: int = 8,
    ) -> Path:
        """Edit an image and save it to `outdir`, return path to file."""
        outdir.mkdir(parents=True, exist_ok=True)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
        ).images[0]

        outfile = outdir / f"edit_{torch.randint(0, 1_000_000, (1,)).item()}.png"
        result.save(outfile)
        return outfile
