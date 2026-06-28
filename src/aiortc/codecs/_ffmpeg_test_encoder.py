from fractions import Fraction
from functools import lru_cache

import av

_CANDIDATE_PIX_FMTS = ("yuv420p", "nv12")


def ffmpeg_test_encoder(encoder, parameters=None):
    if parameters is None:
        parameters = {}

    options = tuple(sorted((str(k), str(v)) for k, v in parameters.items()))
    return _test_encoder(encoder, options)


@lru_cache
def _test_encoder(encoder, options=()):
    # Is the encoder name known to this libavcodec build at all?
    try:
        av.CodecContext.create(encoder, "w")
    except (av.FFmpegError, ValueError, LookupError):
        return False

    option_dict = {key: value for key, value in options}
    width = height = 256
    for pix_fmt in _CANDIDATE_PIX_FMTS:
        try:
            codec = av.CodecContext.create(encoder, "w")
            codec.width = width
            codec.height = height
            codec.pix_fmt = pix_fmt
            codec.time_base = Fraction(1, 30)
            codec.bit_rate = 1_000_000
            if option_dict:
                codec.options = option_dict

            # Encode one defined frame; never hand an encoder uninitialized memory.
            frame = av.VideoFrame(width, height, pix_fmt)
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts = 0
            frame.time_base = Fraction(1, 30)
            codec.encode(frame)
            codec.encode(None)
        except (av.FFmpegError, ValueError):
            continue
        return True

    return False
