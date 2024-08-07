import cv2
import numpy
from PIL import Image


def image_process(image: Image, target_size: int) -> numpy.ndarray:
    # make alpha to white
    image = image.convert('RGBA')
    new_image = Image.new('RGBA', image.size, 'WHITE')
    new_image.alpha_composite(image)
    image = new_image.convert('RGB')
    del new_image

    # Pad image to square
    original_size = image.size
    desired_size = max(max(original_size), target_size)

    delta_width = desired_size - original_size[0]
    delta_height = desired_size - original_size[1]
    top_padding, bottom_padding = delta_height // 2, delta_height - (delta_height // 2)
    left_padding, right_padding = delta_width // 2, delta_width - (delta_width // 2)

    # Convert image data to numpy float32 data
    image = numpy.asarray(image)

    padded_image = cv2.copyMakeBorder(
        src=image,
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # WHITE
    )

    print(padded_image.shape)

    # USE INTER_AREA downscale
    if padded_image.shape[0] > target_size:
        padded_image = cv2.resize(
            src=padded_image,
            dsize=(target_size, target_size),
            interpolation=cv2.INTER_AREA
        )

    # USE INTER_LANCZOS4 upscale
    elif padded_image.shape[0] < target_size:
        padded_image = cv2.resize(
            src=padded_image,
            dsize=(target_size, target_size),
            interpolation=cv2.INTER_LANCZOS4
        )

    # From PIL RGB to OpenCV GBR
    padded_image = padded_image[:, :, ::-1]
    padded_image = padded_image.astype(numpy.float32)

    return padded_image
