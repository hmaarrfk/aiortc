import os
import fractions
import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import tee
from struct import pack, unpack_from
from typing import Optional, Type, TypeVar, cast

import av
from av.frame import Frame
from av.packet import Packet
from av.video.codeccontext import VideoCodecContext

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder, apply_frame_color_properties

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 3_000_000  # 3 Mbps
MIN_BITRATE = 1_000_000  # 1 Mbps
MAX_BITRATE = 30_000_000  # 30 Mbps

MAX_FRAME_RATE = 30
PACKET_MAX = 1100

# HEVC NAL unit types for RTP payload format (RFC 7798)
NAL_TYPE_FU = 49  # Fragmentation Unit
NAL_TYPE_AP = 48  # Aggregation Packet (STAP-A)

NAL_HEADER_SIZE = 2  # HEVC uses 2-byte NAL unit header
FU_HEADER_SIZE = 3  # FU indicator (1) + FU header (2)
LENGTH_FIELD_SIZE = 2
AP_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE

DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="HEVCPayloadDescriptor")
T = TypeVar("T")


def pairwise(iterable: Sequence[T]) -> Iterator[tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class HEVCPayloadDescriptor:
    def __init__(self, first_fragment: bool) -> None:
        self.first_fragment = first_fragment

    def __repr__(self) -> str:
        return f"HEVCPayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> tuple[DESCRIPTOR_T, bytes]:
        output = bytes()

        # HEVC NAL unit header (2 bytes)
        if len(data) < 3:
            raise ValueError("HEVC NAL unit is too short")

        # First byte: F(1) + Type(6) + LayerID(6)
        # For RTP, we extract type from lower 6 bits
        nal_type = (data[0] >> 1) & 0x3F
        f_nri = data[0] & 0x81  # F bit and reserved bits
        layer_id = ((data[0] & 0x01) << 5) | ((data[1] >> 3) & 0x1F)
        pos = NAL_HEADER_SIZE

        if nal_type < 32:
            # single NAL unit (types 0-31)
            output = bytes([0, 0, 0, 1]) + data
            obj = cls(first_fragment=True)
        elif nal_type == NAL_TYPE_FU:
            # fragmentation unit
            if len(data) < pos + 2:
                raise ValueError("HEVC FU-A header is truncated")

            # FU header: S(1) + E(1) + Type(6) + reserved
            fu_header = data[pos]
            start_bit = (fu_header >> 7) & 0x01
            end_bit = (fu_header >> 6) & 0x01
            original_nal_type = fu_header & 0x3F
            pos += 1

            # Second byte of FU header (layer info)
            fu_header2 = data[pos]
            pos += 1

            first_fragment = bool(start_bit)

            if first_fragment:
                # Reconstruct original NAL unit header
                original_nal_header = bytes([
                    (f_nri & 0x81) | ((original_nal_type << 1) & 0xFE) | (layer_id >> 5),
                    ((layer_id << 3) & 0xF8) | (data[1] & 0x07)
                ])
                output += bytes([0, 0, 0, 1])
                output += original_nal_header
            output += data[pos:]

            obj = cls(first_fragment=first_fragment)
        elif nal_type == NAL_TYPE_AP:
            # aggregation packet (STAP-A)
            offsets = []
            while pos < len(data):
                if len(data) < pos + LENGTH_FIELD_SIZE:
                    raise ValueError("HEVC AP length field is truncated")
                nalu_size = unpack_from("!H", data, pos)[0]
                pos += LENGTH_FIELD_SIZE
                offsets.append(pos)

                pos += nalu_size
                if len(data) < pos:
                    raise ValueError("HEVC AP data is truncated")

            offsets.append(len(data) + LENGTH_FIELD_SIZE)
            for start, end in pairwise(offsets):
                end -= LENGTH_FIELD_SIZE
                output += bytes([0, 0, 0, 1])
                output += data[start:end]

            obj = cls(first_fragment=True)
        else:
            raise ValueError(f"HEVC NAL unit type {nal_type} is not supported")

        return obj, output


class HEVCDecoder(Decoder):
    def __init__(self) -> None:
        self.codec = av.CodecContext.create("hevc", "r")

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        try:
            packet = av.Packet(encoded_frame.data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            return cast(list[Frame], self.codec.decode(packet))
        except av.FFmpegError as e:
            logger.warning(
                "HEVCDecoder() failed to decode, skipping package: " + str(e)
            )
            return []


class HEVCEncoder(Encoder):
    def __init__(self, parameters=None) -> None:
        if parameters is None:
            parameters = {}

        if (profile_id := parameters.get('profile-id', '1')) != '1':
            raise ValueError(f"Profile ID {profile_id} is not supported, must be '1'")

        self.__parameters = parameters
        from ._ffmpeg_test_encoder import ffmpeg_test_encoder
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None

        self.__needs_reconfigure: bool = False
        self.__encoder: Optional[str] = None
        self.__target_bitrate: Optional[int] = None
        self.codec: Optional[VideoCodecContext] = None

        selected_encoder = None
        for encoder in [
            "hevc_qsv",
            # prefer QSV???
            "hevc_nvenc",
            "hevc_videotoolbox",
            "libx265",
        ]:
            try:
                if ffmpeg_test_encoder(encoder):
                    selected_encoder = encoder
                    break
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        if not selected_encoder:
            raise RuntimeError(
                "No HEVC encoder available "
                "(tested: hevc_qsv, hevc_nvenc, hevc_videotoolbox, libx265)"
            )

        self.__encoder = selected_encoder
        try:
            self._reset_encoder_settings()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._reset_encoder_settings()

    def _reset_encoder_settings(self) -> None:
        profile_id = self.__parameters.get('profile-id', '1')
        if profile_id == '1':
            profile = 'main'
        elif profile_id == '2':
            profile = 'main10'
        else:
            raise ValueError(f"Profile ID {profile} is not supported, must be '1' or '2'")

        level_id = self.__parameters.get('level-id', '186')
        if level_id == '186':
            level = '6.2'
        else:
            raise ValueError(f"Level ID {level_id} is not supported, must be '186'")

        tier_id = self.__parameters.get('level-id', '0')
        if tier_id == '1':
            tier = 'high'
        else:
            tier = 'main'

        self.__codec_profile = profile

        if self.__encoder == "hevc_qsv":
            av.logging.set_level(av.logging.VERBOSE)
            self.__pix_fmt = "nv12"
            if self.__target_bitrate is None:
                self.__target_bitrate = 1_000_000
            print(f"{self.__target_bitrate:=}")
            self.__codec_options = {
                # I feel like hevc uses these numbers without a period
                # "level": level_id.replace('.', ''),
                "level": "51",  # or 153
                "async_depth": "1",
                "bf": "0",
                'rc': 'cbr',
                "b_strategy": "0",
                "forced_idr": "1",
                # idr_interval is 0 for all frames should be IDR (and not CRA)
                # Streaming decoders have a hard time decoding CRA frames.
                "idr_interval": "0",
                "p_strategy": "0",
                "adaptive_i": "0",
                "adaptive_b": "0",
                "extbrc": "0",
                "look_ahead": "0",
                "low_delay_brc": "1",
                "strict_gop": "1",
                "low_power": "0",
                "gpb": "0",
                "g": "100",
                # "closed_gop": "1",
                'b': str(self.target_bitrate),
                'maxrate': str(self.target_bitrate),
                'minrate': str(self.target_bitrate),
                "profile": profile,
                'tier': tier,
            }
        elif self.__encoder == "hevc_nvenc":
            self.__pix_fmt = "yuv420p"
            if self.__target_bitrate is None:
                self.__target_bitrate = 3_000_000
            self.__codec_options = {
                "level": level,
                "tier": tier,
                "tune": "ull",
                # "rc": "cbr_ld_hq",
                'rc': 'cbr',
                'multipass': 'disabled',
                "preset": "p1",
                'b': str(self.target_bitrate),
                'maxrate': str(int(self.target_bitrate * 3)),
                'minrate': str(int(self.target_bitrate / 3)),
                'profile': profile,
                'bf': '0',
                'b_adapt': '0',
                'rc-lookahead': '0',
                'lookahead_level': '0',
                'b_ref_mode': '0',
                '2pass': '0',
                'no-scenecut': '1',
                'strict_gop': '1',
                'forced-idr': '1',
                'zerolatency': '1',

                "g": "100",                 # shorter GOP keeps IDR cadence tight
                "delay": "0",
                "vbv_bufsize": str(self.target_bitrate // 2),  # critical for NVENC latency
                "async_depth": "1",        # one-frame pipeline depth
            }
        elif self.__encoder == "hevc_videotoolbox":
            # Apple VideoToolbox hardware HEVC encoder, configured for
            # low-latency real-time streaming. VideoToolbox does not expose
            # explicit B-frame / lookahead / rate-control knobs; the encoder
            # manages those internally. The options used here are:
            #   constant_bit_rate=1 -> request constant bitrate (CBR); this
            #                          maps to
            #                          kVTCompressionPropertyKey_ConstantBitRate
            #                          and requires macOS 13 or newer.
            #   realtime=1          -> request real-time (low-latency) encoding.
            #   prio_speed=1        -> prioritize encoding speed over quality.
            #   max_ref_frames=1    -> limit the number of reference frames.
            self.__pix_fmt = "yuv420p"
            # No explicit profile/level is set: videotoolboxenc.c only forwards
            # kVTCompressionPropertyKey_ProfileLevel to the encoder when a
            # profile (or level) is requested, so leaving both unset lets the
            # encoder choose them from the resolution and bitrate. The None
            # profile is skipped in _encode_frame().
            self.__codec_profile = None
            if self.__target_bitrate is None:
                self.__target_bitrate = 3_000_000
            bitrate = self.__target_bitrate
            self.__codec_options = {
                "constant_bit_rate": "1",
                "realtime": "1",
                "prio_speed": "1",
                "max_ref_frames": "1",
                "bf": "0",
                "b": str(bitrate),
                "maxrate": str(int(bitrate * 1.5)),
            }
        elif self.__encoder == "libx265":
            self.__pix_fmt = "yuv420p"
            self.__codec_options = {
                "level": level,
                'profile': profile,
                "tier": tier,
                "tune": "zerolatency",
                "x265-params": "keyint=30:min-keyint=30:scenecut=0",
            }
            if self.__target_bitrate is None:
                self.__target_bitrate = 1_000_000
        else:
            self.__pix_fmt = "yuv420p"
            self.__codec_options = {}
            if self.__target_bitrate is None:
                self.__target_bitrate = DEFAULT_BITRATE

    @staticmethod
    def _packetize_fu(data: bytes) -> list[bytes]:
        available_size = PACKET_MAX - FU_HEADER_SIZE
        payload_size = len(data) - NAL_HEADER_SIZE
        num_packets = math.ceil(payload_size / available_size)
        num_larger_packets = payload_size % num_packets
        package_size = payload_size // num_packets

        # Extract NAL unit header info
        nal_header_byte1 = data[0]
        nal_header_byte2 = data[1]
        nal_type = (nal_header_byte1 >> 1) & 0x3F
        f_nri = nal_header_byte1 & 0x81
        layer_id = ((nal_header_byte1 & 0x01) << 5) | ((nal_header_byte2 >> 3) & 0x1F)

        # FU indicator (2 bytes): F(1) + Type(6) + LayerID(6) + TID(3) + reserved(3)
        # Type is set to NAL_TYPE_FU (49)
        fu_indicator = bytes([
            (f_nri & 0x81) | ((NAL_TYPE_FU << 1) & 0xFE) | (layer_id >> 5),
            ((layer_id << 3) & 0xF8) | (nal_header_byte2 & 0x07)
        ])

        # FU header (1 byte): S(1) + E(1) + Type(6)
        fu_header_end = (nal_type & 0x3F) | 0x40  # E bit set
        fu_header_middle = nal_type & 0x3F
        fu_header_start = (nal_type & 0x3F) | 0x80  # S bit set
        fu_header = fu_header_start

        packages = []
        offset = NAL_HEADER_SIZE
        while offset < len(data):
            if num_larger_packets > 0:
                num_larger_packets -= 1
                payload = data[offset : offset + package_size + 1]
                offset += package_size + 1
            else:
                payload = data[offset : offset + package_size]
                offset += package_size

            if offset == len(data):
                fu_header = fu_header_end

            packages.append(fu_indicator + bytes([fu_header]) + payload)

            fu_header = fu_header_middle
        assert offset == len(data), "incorrect fragment data"

        return packages

    @staticmethod
    def _packetize_ap(
        data: bytes, packages_iterator: Iterator[bytes]
    ) -> tuple[bytes, bytes]:
        counter = 0
        available_size = PACKET_MAX - AP_HEADER_SIZE

        # AP header starts with FU indicator (NAL_TYPE_AP)
        ap_header_byte1 = data[0]
        ap_header_byte2 = data[1]
        f_nri = ap_header_byte1 & 0x81
        layer_id = ((ap_header_byte1 & 0x01) << 5) | ((ap_header_byte2 >> 3) & 0x1F)

        ap_header = bytes([
            (f_nri & 0x81) | ((NAL_TYPE_AP << 1) & 0xFE) | (layer_id >> 5),
            ((layer_id << 3) & 0xF8) | (ap_header_byte2 & 0x07)
        ])

        payload = bytes()
        try:
            nalu = data  # with header
            while len(nalu) <= available_size and counter < 9:
                # Update F and NRI bits from aggregated NAL units
                nalu_f_nri = nalu[0] & 0x81
                if (ap_header[0] & 0x80) == 0 and (nalu_f_nri & 0x80) != 0:
                    ap_header = bytes([ap_header[0] | 0x80, ap_header[1]])

                nri = nalu[0] & 0x60
                if (ap_header[0] & 0x60) < nri:
                    ap_header = bytes([(ap_header[0] & 0x9F) | nri, ap_header[1]])

                available_size -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(packages_iterator)

            if counter == 0:
                nalu = next(packages_iterator)
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        else:
            return ap_header + payload, nalu

    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        # HEVC uses the same start code pattern as H.264
        i = 0
        while True:
            # Find the start of the NAL unit.
            #
            # NAL Units start with the 3-byte start code 0x000001 or
            # the 4-byte start code 0x00000001.
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                return

            # Jump past the start code
            i += 3
            nal_start = i

            # Find the end of the NAL unit (end of buffer OR next start code)
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                yield buf[nal_start : len(buf)]
                return
            elif buf[i - 1] == 0:
                # 4-byte start code case, jump back one byte
                yield buf[nal_start : i - 1]
            else:
                yield buf[nal_start:i]

    @classmethod
    def _packetize(cls, packages: Iterable[bytes]) -> list[bytes]:
        packetized_packages = []

        packages_iterator = iter(packages)
        package = next(packages_iterator, None)
        while package is not None:
            if len(package) > PACKET_MAX:
                packetized_packages.extend(cls._packetize_fu(package))
                package = next(packages_iterator, None)
            else:
                packetized, package = cls._packetize_ap(package, packages_iterator)
                packetized_packages.append(packetized)

        return packetized_packages

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        if self.codec and (
            frame.width != self.codec.width
            or self.encoder != self.codec.name
            or frame.height != self.codec.height
            or self.__needs_reconfigure
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            # force a complete image
            frame.pict_type = av.video.frame.PictureType.I
        else:
            # reset the picture type, otherwise no B-frames are produced
            frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            self.__needs_reconfigure = False
            try:
                os.environ["LIBVA_MESSAGING_LEVEL"] = os.environ.get("LIBVA_MESSAGING_LEVEL", "1")
                self.codec = av.CodecContext.create(self.encoder, "w")
                self.codec.width = frame.width
                self.codec.height = frame.height
                self.codec.bit_rate = self.target_bitrate
                self.codec.pix_fmt = self.pix_fmt
                self.codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
                self.codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
                self.codec.options = self.codec_options
                apply_frame_color_properties(self.codec, frame)
                # codec_profile may be None (e.g. VideoToolbox); skip the
                # assignment so the encoder chooses the profile itself.
                if self.codec_profile is not None:
                    self.codec.profile = self.codec_profile
            except Exception as e:
                print(f"[HEVC DEBUG] ERROR setting up encoder: {e}")
                raise e

        try:
            data_to_send = b"".join(
                bytes(package)
                for package in self.codec.encode(frame)
            )
        except Exception as e:
            print(f"[HEVC DEBUG] ERROR encoding frame: {e}")
            raise e

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        packages = self._encode_frame(frame, force_keyframe)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        assert isinstance(packet, av.Packet)
        packages = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    @property
    def target_bitrate(self) -> int:
        """
        Target bitrate in bits per second.
        """
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        # we only adjust bitrate if it changes by over 5%
        if abs(bitrate - self.__target_bitrate) > 0.05 * self.__target_bitrate:
            self.__needs_reconfigure = True

        self.__target_bitrate = bitrate
        self.__codec_options['b'] = str(bitrate)
        self.__codec_options['maxrate'] = str(int(bitrate * 3))
        self.__codec_options['minrate'] = str(int(bitrate / 3))

    @property
    def encoder(self) -> Optional[str]:
        return self.__encoder

    @encoder.setter
    def encoder(self, value: str) -> None:
        from ._ffmpeg_test_encoder import ffmpeg_test_encoder
        if not ffmpeg_test_encoder(value):
            raise ValueError(f"Encoder {value} is not available")

        old_encoder = self.__encoder
        self.__encoder = value

        if value != old_encoder:
            self.__needs_reconfigure = True
            self._reset_encoder_settings()

    @property
    def pix_fmt(self) -> str:
        return self.__pix_fmt

    @pix_fmt.setter
    def pix_fmt(self, value: str) -> None:
        if value != self.__pix_fmt:
            self.__needs_reconfigure = True
        self.__pix_fmt = value

    @property
    def codec_profile(self) -> str:
        return self.__codec_profile

    @codec_profile.setter
    def codec_profile(self, value: str) -> None:
        if value != self.__codec_profile:
            self.__needs_reconfigure = True

        self.__codec_profile = value

    @property
    def codec_options(self) -> dict[str, str]:
        return self.__codec_options

    @codec_options.setter
    def codec_options(self, value: dict[str, str]) -> None:
        if value != self.__codec_options:
            self.__needs_reconfigure = True
        self.__codec_options = value


def hevc_depayload(payload: bytes) -> bytes:
    descriptor, data = HEVCPayloadDescriptor.parse(payload)
    return data
