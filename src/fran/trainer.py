from pathlib import Path
from PIL import Image
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ReAgingDataset(Dataset):
    def __init__(self, image_path: str):
        self.image_path = sorted(list(Path(image_path).glob("*")))
        self.age_list = list(range(20, 90, 5))
        self.image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                 transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                                 transforms.Resize((512, 512))
                              ]
            )
        self.mask_transform = transforms.Compose([
                                transforms.ToTensor(),
                                 transforms.Resize((512, 512))
                              ]
            )
        

    def __getitem__(self, index):
        
        src_age = self.age_list[random.randint(0,13)]
        target_age = self.age_list[random.randint(0,13)]

        image = Image.open(self.image_path[index] / f"{src_age}.png")
        label = Image.open(self.image_path[index] / f"{target_age}.png")
        mask = Image.open(self.image_path[index] / "mask.png")

        image = self.image_transform(image)
        label = self.image_transform(label)
        mask = self.mask_transform(mask)

        src_age_mask = mask * src_age
        target_age_mask = mask * target_age

        image_with_mask = torch.cat([image, src_age_mask, target_age_mask], axis=1)

        return dict(image=image_with_mask, mask=mask, label=label)


class Trainer:

    def __init__(self,
                 image_path,
                 model,
                 ):
        
        raise NotImplemented