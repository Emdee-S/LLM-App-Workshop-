from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(state_path: str, meta_path: str, device: str = "cpu") -> Tuple[nn.Module, transforms.Compose, List[str], torch.device]:
    target = torch.device(device)
    meta = torch.load(meta_path, map_location=target)
    classes: List[str] = list(meta.get("classes", [str(i) for i in range(10)]))
    model = CifarCNN(num_classes=len(classes)).to(target)
    state = torch.load(state_path, map_location=target)
    model.load_state_dict(state, strict=True)
    model.eval()
    normalize = transforms.Normalize(
        tuple(meta.get("normalize_mean", (0.4914, 0.4822, 0.4465))),
        tuple(meta.get("normalize_std", (0.2470, 0.2435, 0.2616))),
    )
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])
    return model, preprocess, classes, target


@torch.no_grad()
def predict_image(image_path: str, model: nn.Module, preprocess: transforms.Compose, classes: List[str], device: torch.device, topk: int = 5):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk = min(topk, len(classes))
    top_probs, top_indices = torch.topk(probs, k=topk)
    return [(classes[i.item()], top_probs[j].item()) for j, i in enumerate(top_indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 inference with saved weights")
    parser.add_argument("--state", default="cifar10_cnn_state.pt", help="Path to model state dict")
    parser.add_argument("--meta", default="cifar10_meta.pt", help="Path to metadata file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cpu", help="Device to run on, e.g. cpu or cuda:0")
    parser.add_argument("--topk", type=int, default=5, help="Show top-K predictions")
    args = parser.parse_args()

    model, preprocess, classes, device = load_model(args.state, args.meta, device=args.device)
    results = predict_image(args.image, model, preprocess, classes, device, topk=args.topk)

    print("Top predictions:")
    for label, p in results:
        print(f"{label}: {p:.3f}")


if __name__ == "__main__":
    main()


