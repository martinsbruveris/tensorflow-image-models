from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds

from tfimm.train.registry import cfg_serializable


@dataclass
class TFDSConfig:
    dataset_name: str
    split: str
    input_size: tuple
    batch_size: int
    repeat: bool = False
    shuffle: bool = False
    nb_samples: int = None
    cache: bool = False
    dtype: str = "float32"


@cfg_serializable
class TFDSWrapper:
    """Wrapper class around tensorflow datasets."""

    cfg_class = TFDSConfig

    def __init__(self, cfg: TFDSConfig):
        self.cfg = cfg
        self.root_ds = tfds.load(
            self.cfg.dataset_name,
            split=self.cfg.split,
            shuffle_files=self.cfg.shuffle,
            as_supervised=True,
            # We disable auto-caching at this point, because if we use `nb_samples`,
            # TF will print repeated warnings that we are not exhausting the iterator
            # of a cached dataset. Use `cfg.cache` instead, which caches the dataset
            # after applying `nb_samples`.
            read_config=tfds.ReadConfig(try_autocache=False),
        )

    def get_ds(self):
        ds = self.root_ds
        if self.cfg.nb_samples:
            ds = ds.take(self.cfg.nb_samples)
        if self.cfg.cache:
            ds = ds.cache()
        if self.cfg.shuffle:
            ds = ds.shuffle(buffer_size=3000)
        if self.cfg.repeat:
            ds = ds.repeat()
        if self.cfg.batch_size:
            ds = ds.batch(self.cfg.batch_size, drop_remainder=False)
        ds = ds.map(self.decode_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def decode_batch(self, img, label):
        if self.cfg.input_size:
            img = tf.image.resize(img, size=self.cfg.input_size)

        return img, label

    def __iter__(self):
        return iter(self.get_ds())
