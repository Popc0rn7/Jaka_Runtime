import numpy as np
from PIL import Image
from pathlib import Path


def load_static_rgb_image(path: Path, height: int, width: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize to camera shape."""
    if not path.is_file():
        raise FileNotFoundError(f"Static camera image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        target_ratio = width / height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = round(source_height * target_ratio)
            left = (source_width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, source_height))
        elif source_ratio < target_ratio:
            crop_height = round(source_width / target_ratio)
            top = (source_height - crop_height) // 2
            image = image.crop((0, top, source_width, top + crop_height))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)
