import torch
from dreamsim import dreamsim
from PIL import Image
from torchvision.transforms import v2 as T

_ = torch.set_grad_enabled(False)
torchdev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(torchdev)
torch.set_float32_matmul_precision("high")


img_size = 224
t = T.Compose(
    [
        T.ToImage(),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToDtype(torch.float32, scale=True),
    ]
)

# Load model
model, preprocess = dreamsim(pretrained=True, device=torchdev, cache_dir="data/dreamsim")

model = model.eval().to(torchdev)

# Load images
img_ref = preprocess(Image.open("repos/dreamsim/images/ref_1.png")).to(torchdev)
img_0 = preprocess(Image.open("repos/dreamsim/images/img_a_1.png")).to(torchdev)
img_1 = preprocess(Image.open("repos/dreamsim/images/img_b_1.png")).to(torchdev)

# Get distance
d0 = model(img_ref, img_0)
d1 = model(img_ref, img_1)

print(f"d0: {d0.item():.8f}, d1: {d1.item():.8f}")
