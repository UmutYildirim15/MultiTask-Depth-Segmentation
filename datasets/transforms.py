import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_geometric_transform(image_size: int, train: bool) -> A.Compose:
    ops = [A.Resize(image_size, image_size)]
    if train:
        ops.append(A.HorizontalFlip(p=0.5))
    return A.Compose(ops, additional_targets={"depth": "image"})


def get_photometric_transform(train: bool) -> A.Compose:
    ops = []
    if train:
        ops.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3))
        ops.append(A.GaussNoise(std_range=(0.02, 0.05), p=0.2))
        ops.append(A.MotionBlur(blur_limit=5, p=0.2))
    ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())
    return A.Compose(ops)


def apply_transforms(image, mask, depth, image_size: int, train: bool):
    geo = get_geometric_transform(image_size, train)
    geo_out = geo(image=image, mask=mask, depth=depth)

    photo = get_photometric_transform(train)
    photo_out = photo(image=geo_out["image"])

    return photo_out["image"], geo_out["mask"], geo_out["depth"]