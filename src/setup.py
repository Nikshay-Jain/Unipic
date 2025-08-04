import os, torch
from torchvision.models import mobilenet_v3_small

def setup(save_path="mobilenet_v3_feat.pt"):
    model = mobilenet_v3_small(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(save_path)
    print(f"Saved TorchScript model to: {save_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    setup("models/mobilenet_v3_feat.pt")