# Since starting up ffmpeg can be time consuming, we use lru_cache
# to remember the results of the tests.
# If shape is provided as a tuple, it is something
# that can be hashed by lru_cache in order to  ensure
# the function returns quickly the second time it is requested.
from curses import keyname
from functools import lru_cache
import subprocess


def ffmpeg_test_encoder(
    encoder,
    parameters=None,
):
    if parameters is None:
        parameters = {}

    extra_codec_arguments = ()
    for key, value in parameters.items():
        extra_codec_arguments += (f'-{key}:v', str(value))

    return _ffmpeg_test_encoder(encoder, extra_codec_arguments)

@lru_cache
def _ffmpeg_test_encoder(encoder, extra_codec_arguments=None):
    if extra_codec_arguments is None:
        extra_codec_arguments = ()
    # Note that images smaller than 256 x 256 may not be compatible
    # with all encoders
    shape = (256, 256)
    # Use the null streams to validate if we can encode anything
    # https://trac.ffmpeg.org/wiki/Null
    # This effecitevely runs
    # ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256:d=8 -f null -vcodec h264_nvenc -
    cmd = [
        "ffmpeg", "-hide_banner",
        "-f", "lavfi",
        # python works in height x width
        # but ffmpeg expects width x height
        # this makes a different for small videos with h264_nvenc
        "-i", f"nullsrc=s={shape[1]}x{shape[0]}:d=8",
        "-vcodec", encoder,
    ] + list(extra_codec_arguments) + [
        "-f", "null",
        "-",
    ]
    print(cmd)
    p = subprocess.run(
        cmd,
        stdin=subprocess.PIPE,
        capture_output=True,
        check=False,
    )
    return p.returncode == 0
