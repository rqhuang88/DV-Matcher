import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm
# from featup.plotting import plot_feats, plot_lang_heatmaps

input_size = 224
image_path = "FeatUp/sample-images/plant.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True

transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])

image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
hr_feats = upsampler(image_tensor)
lr_feats = upsampler.model(image_tensor)
print(lr_feats.shape)
# plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0])