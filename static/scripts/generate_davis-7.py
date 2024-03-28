import glob
import os

from absl import app
from absl import flags
import cv2
import mediapy
import numpy as np


_DAVIS_PATH = flags.DEFINE_string(
    'davis_source_path', None, 'Path to the original DAVIS dataset.'
)

_DAVIS7_PATH = flags.DEFINE_string(
    'davis7_target_path', None, 'Path where to create DAVIS-7.'
)

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_GAMMA = 2.2
TARGET_WIDTH = 256
FRAME_STEP = 1
NUM_IMAGES_PER_EXAMPLE = 9


def _resample_image(image: np.ndarray, resample_image_size: int) -> np.ndarray:
  """Re-samples and returns an `image` to be `resample_image_size`."""
  # Convert image from uint8 gamma [0..255] to float linear [0..1].
  image = image.astype(np.float32) / _UINT8_MAX_F
  image = np.power(np.clip(image, 0, 1), _GAMMA)

  # Re-size the image
  new_size = (resample_image_size, resample_image_size)
  image = cv2.resize(
      image,
      new_size,
      interpolation=cv2.INTER_AREA
      if new_size[0] <= image.shape[0] else cv2.INTER_CUBIC)

  # Convert back from float linear [0..1] to uint8 gamma [0..255].
  image = np.power(np.clip(image, 0, 1), 1.0 / _GAMMA)
  image = np.clip(image * _UINT8_MAX_F + 0.5, 0.0,
                  _UINT8_MAX_F).astype(np.uint8)
  return image


def _central_crop_to_square(image):
  half = min(image.shape[0], image.shape[1])//2
  image = image[image.shape[0]//2-half:image.shape[0]//2+half,
                image.shape[1]//2-half:image.shape[1]//2+half]
  return image


def preprocess_image(image):
  image = _central_crop_to_square(image)
  image = _resample_image(image, TARGET_WIDTH)
  return image


def main(argv):
  del argv
  example_idx = 0
  for shot_dir in glob.glob(
      os.path.join(_DAVIS_PATH.value, 'JPEGImages/480p', '*')
  ):
    print('Processing shot dir: ', shot_dir)
    frames = sorted(glob.glob(os.path.join(shot_dir, '*.jpg')))
    frames = frames[::FRAME_STEP]
    n = NUM_IMAGES_PER_EXAMPLE
    frame_chunks = [frames[start:start+n] for start in range(0, len(frames), n)]
    for frame_chunk in frame_chunks:
      if len(frame_chunk) == NUM_IMAGES_PER_EXAMPLE:
        nlet_dict = {
            f'frame_{idx:03}': frame_path
            for idx, frame_path in enumerate(frame_chunk)
        }
        for frame_name, frame_path in nlet_dict.items():
          image = mediapy.read_image(frame_path)
          image = preprocess_image(image)
          target_path = os.path.join(
              _DAVIS7_PATH.value, str(example_idx).zfill(4), frame_name + '.png'
          )
          if not os.path.isdir(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path))
          mediapy.write_image(target_path, image)
        example_idx += 1

  if example_idx != 357:
    print(
        'Something went wrong. Generated an unexpected number of examples: ',
        example_idx,
    )
  else:
    print('Generated {example_idx} examples as expected.')


if __name__ == "__main__":
  app.run(main)
