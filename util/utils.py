import torch
from torchvision.transforms import v2 as T


def get_preprocess(model_type):
    if "lpips" in model_type:
        return "LPIPS"
    elif "dists" in model_type:
        return "DISTS"
    elif "psnr" in model_type:
        return "PSNR"
    elif "ssim" in model_type:
        return "SSIM"
    elif "clip" in model_type or "open_clip" in model_type or "dino" in model_type or "mae" in model_type:
        return "DEFAULT"
    else:
        return "DEFAULT"


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = T.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.0
    else:
        if preprocess == "DEFAULT":
            t = T.Compose(
                [
                    T.ToImage(),
                    T.Resize((load_size, load_size), interpolation=interpolation),
                    T.ToDtype(torch.float32, scale=True),
                ]
            )
        elif preprocess == "DISTS":
            t = T.Compose(
                [
                    T.ToImage(),
                    T.Resize((256, 256), interpolation=interpolation),
                    T.ToDtype(torch.float32, scale=True),
                ]
            )
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = T.ToTensor()
        else:
            raise ValueError("Unknown preprocessing method")
        return lambda pil_img: t(pil_img.convert("RGB"))
