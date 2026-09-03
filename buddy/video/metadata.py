"""Small dependency-free readers for local video container metadata."""

import struct
from pathlib import Path
from typing import BinaryIO, Iterator


def detect_video_duration(path: str | Path) -> float | None:
    """Read MP4/MOV movie duration, returning None for unsupported containers."""
    video_path = Path(path)
    if video_path.suffix.casefold() not in {".mp4", ".mov", ".m4v"}:
        return None
    try:
        with video_path.open("rb") as stream:
            file_end = video_path.stat().st_size
            for box_type, payload_start, box_end in _boxes(stream, 0, file_end):
                if box_type != b"moov":
                    continue
                for child_type, child_start, child_end in _boxes(
                    stream, payload_start, box_end
                ):
                    if child_type == b"mvhd":
                        return _read_mvhd_duration(stream, child_start, child_end)
    except (OSError, ValueError, struct.error):
        return None
    return None


def _boxes(
    stream: BinaryIO,
    start: int,
    end: int,
) -> Iterator[tuple[bytes, int, int]]:
    position = start
    while position + 8 <= end:
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8:
            return
        size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                return
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ValueError("Invalid ISO BMFF box size.")
        box_end = position + size
        yield box_type, position + header_size, box_end
        position = box_end


def _read_mvhd_duration(stream: BinaryIO, start: int, end: int) -> float | None:
    stream.seek(start)
    payload = stream.read(min(end - start, 32))
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 0:
        timescale = struct.unpack_from(">I", payload, 12)[0]
        duration = struct.unpack_from(">I", payload, 16)[0]
    elif version == 1 and len(payload) >= 32:
        timescale = struct.unpack_from(">I", payload, 20)[0]
        duration = struct.unpack_from(">Q", payload, 24)[0]
    else:
        return None
    if timescale == 0:
        return None
    return duration / timescale
