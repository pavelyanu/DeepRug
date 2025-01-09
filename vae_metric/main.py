from PIL import Image
from PIL.ImageFile import ImageFile
import time

import torch
from torchvision import transforms

from metric import Metric

def load_image(path: str) -> ImageFile:
    return Image.open(path)

def image_to_tensor(image: ImageFile) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)

def load_image_tensor(path: str) -> torch.Tensor:
    return image_to_tensor(load_image(path))

def main():
    model_path = "/home/pavel/dev/university/masters/1-winter/seminar_applications/DeepRug/results/vqvae_20250109_180050_4900.pth"
    negative_example = "/home/pavel/dev/university/masters/1-winter/seminar_applications/DeepRug/not_carpet.png"
    maybe_example = "/home/pavel/dev/university/masters/1-winter/seminar_applications/DeepRug/carpet_maybe.png"
    positive_example = "/home/pavel/dev/university/masters/1-winter/seminar_applications/DeepRug/carpet.png"
    gae_carpet_example = "/home/pavel/dev/university/masters/1-winter/seminar_applications/DeepRug/gae_carpet.png"

    negative_example = load_image_tensor(negative_example)
    maybe_example = load_image_tensor(maybe_example)
    positive_example = load_image_tensor(positive_example)
    gae_carpet_example = load_image_tensor(gae_carpet_example)

    metric = Metric(model_path)

    start = time.time()
    print(f"Negative example: {metric.reconstruction_loss(negative_example)}")
    end = time.time()
    print(f"Time: {end - start}")
    print()

    start = time.time()
    print(f"Maybe example: {metric.reconstruction_loss(maybe_example)}")
    end = time.time()
    print(f"Time: {end - start}")
    print()

    start = time.time()
    print(f"Positive example: {metric.reconstruction_loss(positive_example)}")
    end = time.time()
    print(f"Time: {end - start}")
    print()

    start = time.time()
    print(f"GAE carpet example: {metric.reconstruction_loss(gae_carpet_example)}")
    end = time.time()
    print(f"Time: {end - start}")
    print()

if __name__ == "__main__":
    main()

