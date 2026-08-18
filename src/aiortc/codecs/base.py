from abc import ABCMeta, abstractmethod

from av.frame import Frame
from av.packet import Packet
from av.video.codeccontext import VideoCodecContext
from av.video.frame import VideoFrame

from ..jitterbuffer import JitterFrame

# FFmpeg spells "unspecified" as 2 for a matrix, primaries or transfer, and 0
# for a range.
AVCOL_SPC_UNSPECIFIED = 2
AVCOL_PRI_UNSPECIFIED = 2
AVCOL_TRC_UNSPECIFIED = 2
AVCOL_RANGE_UNSPECIFIED = 0


def apply_frame_color_properties(
    codec: VideoCodecContext, frame: VideoFrame
) -> None:
    """Carry a frame's declared colour onto the context that will encode it.

    An encoder writes its colour signalling from the context, not from the
    frame, so a sender that fills these in on the frame alone is ignored: the
    stream then says nothing, and a decoder reading nothing supplies its own
    answer -- limited range, BT.709 at HD, and a video transfer curve. Content
    that was none of those, such as full-range sRGB read off a GPU canvas, is
    then altered in every pixel.

    Only fields the frame actually specifies are copied, so a caller that sets
    none of them keeps whatever the encoder would have chosen.

    Used by the H.264 and HEVC encoders, which is where it works: encoded and
    decoded back, both keep all four fields. VP8 deliberately does not call it,
    because it cannot honour it -- its bitstream holds one colour-space bit
    meaning BT.601 and a clamping bit, with nowhere to put a matrix, primaries
    or a transfer curve, and the same round trip comes back BT.601 limited
    whatever it was given. A sender that needs its colour understood has to
    encode BT.601 limited when VP8 is what got negotiated.
    """
    if frame.colorspace != AVCOL_SPC_UNSPECIFIED:
        codec.colorspace = frame.colorspace
    if frame.color_range != AVCOL_RANGE_UNSPECIFIED:
        codec.color_range = frame.color_range
    if frame.color_primaries != AVCOL_PRI_UNSPECIFIED:
        codec.color_primaries = frame.color_primaries
    if frame.color_trc != AVCOL_TRC_UNSPECIFIED:
        codec.color_trc = frame.color_trc


class Decoder(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        pass  # pragma: no cover


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        pass  # pragma: no cover

    @abstractmethod
    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        pass  # pragma: no cover
