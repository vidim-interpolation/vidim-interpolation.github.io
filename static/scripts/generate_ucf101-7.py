import os
from absl import app
from absl import flags
import mediapy as media
import tensorflow_datasets as tfds


_UCF101_7_PATH = flags.DEFINE_string(
    'ucf101_7_target_path', None, 'Path where to create UCF101-7.'
)


def main(argv):
  del argv
  frames_per_example = 9
  output_path = _UCF101_7_PATH.value

  dl_config = tfds.download.DownloadConfig(verify_ssl=False)
  ds = tfds.load(
      'ucf101',
      split='test',
      download_and_prepare_kwargs={'download_config': dl_config},
  )

  def output_example(frames, example_idx):
    example_path = os.path.join(output_path, str(example_idx).zfill(4))
    if not os.path.isdir(example_path):
      os.makedirs(example_path)

    for frame_idx, frame in enumerate(frames):
      path = os.path.join(example_path, f'frame_{str(frame_idx).zfill(3)}.png')
      print(frame.shape)
      media.write_image(path, frame)

  idx = 0
  examples_to_generate = 400
  for ex in ds:
    frames = ex['video'].numpy()
    n = frames_per_example
    frame_chunks = [frames[start:start+n] for start in range(0, len(frames), n)]
    for frame_chunk in frame_chunks:
      if len(frame_chunk) == frames_per_example:
        output_example(frame_chunk, idx)
        idx = idx + 1
        print(idx)
      if idx >= examples_to_generate:
        print('UCF101-7 successfully generated.')
        return


if __name__ == "__main__":
  app.run(main)
