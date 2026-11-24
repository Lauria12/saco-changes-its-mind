import sys
import torch
from saco.models import load_model

checkpoint_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_model("vit_b16", "imagenette", checkpoint_path, device)
print("Loaded OK, family:", bundle.family, "num_classes:", bundle.num_classes)
x = torch.randn(1, 3, 224, 224).to(device)
logits = bundle.forward_logits(x)
print("forward pass OK, logits shape:", logits.shape)
