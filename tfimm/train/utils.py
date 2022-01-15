import logging
from collections import abc
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from urllib.parse import urlparse

try:
    import boto3
except ImportError:
    boto3 = None
    logging.info("Could not import `boto3`. S3 support not available")

# When asked to look for TFRecord files we consider these file suffixes
TFRECORD_SUFFIXES = (".tfrecord", ".tfrecords")


def setup_logging(logging_level=logging.INFO):
    """
    Creates a logger that logs to stdout. Sets this logger as the global default.
    Logging format is
        2020-12-05 21:44:09,908: Message.

    Returns:
        Doesn't return anything. Modifies global logger.
    """
    logging.basicConfig(level=logging_level)
    fmt = logging.Formatter("%(asctime)s: %(message)s", datefmt="%y-%b-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # We want to change the format of the root logger, which is the first one
    # in the logger.handlers list. A bit hacky, but there we go.
    root = logger.handlers[0]
    root.setFormatter(fmt)


def collect_tfrecord_files(data_dir: Union[str, Iterable[str]]) -> List[str]:
    """
    We collect all tfrecord files in the locations specified by data_dir.

    Args:
        data_dir: Where to look for tfrecords. This can be either one file, one
            directory or an iterable of files or directories. If `data_dir` is a
            directory, we recurse through subdirectories. On S3, we look at all subkeys.

    Returns:
        List of tfrecord files, possibly empty.
    """
    return collect_files_with_suffix(data_dir, TFRECORD_SUFFIXES)


def collect_files_with_suffix(
    data_dir: Union[str, List[str]], suffix: Union[str, Tuple[str, ...]]
) -> List[str]:
    """
    We collect all files in the locations specified by data_dir with the given suffixes.

    Args:
        data_dir: Where to look for tfrecords. This can be either one file, one
            directory or an iterable of files or directories. If `data_dir` is a
            directory, we recurse through subdirectories. On S3, we look at all subkeys.
        suffix: Which suffixes are we allowed to match

    Returns:
        List of files, possibly empty.
    """
    matching_files = []
    if data_dir is None:
        return matching_files
    elif isinstance(data_dir, abc.Iterable):
        for d in data_dir:
            matching_files.extend(collect_files_with_suffix(d, suffix))
    else:  # Assume it is either string or Path
        data_dir = str(data_dir)
        if data_dir.startswith("s3://"):
            matching_files = _collect_s3_files(data_dir, suffix)
        else:
            matching_files = _collect_local_files(data_dir, suffix)
    matching_files.sort()
    return list(matching_files)


def _collect_local_files(
    data_dir: str, suffix: Union[str, Tuple[str, ...]]
) -> List[str]:
    data_dir = Path(data_dir)
    if data_dir.suffix in suffix:
        matching_files = [str(data_dir)]
    elif data_dir.is_dir():
        matching_files = [str(f) for s in suffix for f in data_dir.rglob(f"*{s}")]
    else:
        matching_files = []
    return matching_files


def _collect_s3_files(data_dir: str, suffix: Union[str, Tuple[str, ...]]) -> List[str]:
    if data_dir.endswith(suffix):
        return [data_dir]

    bucket, prefix = _split_s3_url(data_dir)
    tfrecord_keys = _get_matching_s3_keys(bucket, prefix=prefix, suffix=suffix)
    tfrecord_files = [f"s3://{bucket}/{key}" for key in tfrecord_keys]
    return tfrecord_files


def _get_matching_s3_objects(
    bucket: str,
    prefix: Union[str, Tuple[str, ...]] = "",
    suffix: Union[str, Tuple[str, ...]] = "",
):
    """
    Generate objects in an S3 bucket.

    bucket: Name of the S3 bucket.
    prefix: Only fetch objects whose key starts with this prefix (optional).
    suffix: Only fetch objects whose keys end with this suffix (optional).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    kwargs = {"Bucket": bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix,)
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj


def _get_matching_s3_keys(
    bucket: str,
    prefix: Union[str, Tuple[str, ...]] = "",
    suffix: Union[str, Tuple[str, ...]] = "",
):
    """
    Generate the keys in an S3 bucket.

    bucket: Name of the S3 bucket.
    prefix: Only fetch keys that start with this prefix (optional).
    suffix: Only fetch keys that end with this suffix (optional).
    """
    for obj in _get_matching_s3_objects(bucket, prefix, suffix):
        yield obj["Key"]


def _split_s3_url(url):
    """
    Source:
    https://stackoverflow.com/questions/42641315/s3-urls-get-bucket-name-and-path
    """
    parsed = urlparse(url, allow_fragments=False)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if parsed.query:
        key += "?" + parsed.query
    return bucket, key
