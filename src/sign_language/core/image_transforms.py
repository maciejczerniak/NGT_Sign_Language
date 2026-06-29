"""Shared image transforms that do not depend on MediaPipe or OpenCV.

Provides the standard ImageNet normalisation pipeline used by EfficientNet-B0
for both training and inference. Keeping transforms here avoids duplication
between the preprocessing and training modules.
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_imagenet_transform(img_size: int) -> transforms.Compose:
    """Create the ImageNet preprocessing transform pipeline used by EfficientNet-B0.

    Resizes the input image to a square of ``img_size × img_size`` pixels,
    converts it to a float tensor, and normalises using ImageNet mean and
    standard deviation values.

    :param img_size: Target image size in pixels applied to both height and width.
    :returns: A :class:`~torchvision.transforms.Compose` pipeline consisting
        of resize, to-tensor, and normalise steps.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )
