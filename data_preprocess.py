
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_without_norm(image_size):
    return transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor()
    ])

def preprocess_hflip(img, normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        F.hflip(img),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])
def preprocess_with_augmentation_without_norm(image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor()
    ])
