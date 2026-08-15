"""Read the JSON policy-running frames written by the BeyondMimic runtime.

The implementation deliberately reads only the small MCAP subset used by the
runtime logger: channel metadata plus chunked JSON messages.  Some historical
runtime files, including the JC01 recording used in this project, omit the
``records`` length field in each Chunk record.  Such a file cannot be read by
the strict MCAP reader although its chunk payload is otherwise valid.  This
reader accepts both the standard layout and that legacy layout, without trying
to be a general-purpose MCAP implementation.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any


_MAGIC = b"\x89MCAP0\r\n"
_OP_CHANNEL = 0x04
_OP_CHUNK = 0x06
_OP_MESSAGE = 0x05


def _read_string(payload: bytes, offset: int) -> tuple[str, int]:
  if offset + 4 > len(payload):
    raise ValueError("Truncated MCAP string length")
  size = struct.unpack_from("<I", payload, offset)[0]
  start = offset + 4
  end = start + size
  if end > len(payload):
    raise ValueError("Truncated MCAP string payload")
  return payload[start:end].decode("utf-8"), end


def _channel_topic(payload: bytes) -> tuple[int, str]:
  if len(payload) < 4:
    raise ValueError("Truncated MCAP Channel record")
  channel_id = struct.unpack_from("<H", payload, 0)[0]
  topic, _ = _read_string(payload, 4)  # Skip schema id; topic is sufficient here.
  return channel_id, topic


def _decode_chunk(payload: bytes) -> bytes:
  """Return uncompressed inner records for a standard or legacy Chunk."""
  if len(payload) < 32:
    raise ValueError("Truncated MCAP Chunk record")
  uncompressed_size = struct.unpack_from("<Q", payload, 16)[0]
  compression_size = struct.unpack_from("<I", payload, 28)[0]
  compression_start = 32
  compression_end = compression_start + compression_size
  if compression_end > len(payload):
    raise ValueError("Truncated MCAP Chunk compression field")
  compression = payload[compression_start:compression_end].decode("utf-8")
  remainder = payload[compression_end:]

  # Standard MCAP stores an unsigned 64-bit records length after the
  # compression string.  The runtime's legacy writer omitted it.  Treat the
  # field as standard only when it exactly describes the remaining bytes.
  encoded_records = remainder
  if len(remainder) >= 8:
    declared_size = struct.unpack_from("<Q", remainder, 0)[0]
    if declared_size == len(remainder) - 8:
      encoded_records = remainder[8:]

  if compression == "":
    decoded = encoded_records
  elif compression == "zstd":
    try:
      import zstandard
    except ImportError as exc:  # pragma: no cover - installation diagnostic.
      raise ImportError(
        "zstandard is required to read this zstd-compressed MCAP. "
        "Install it in the experiment environment (for example `uv sync`)."
      ) from exc
    decoded = zstandard.ZstdDecompressor().decompress(
      encoded_records, max_output_size=int(uncompressed_size),
    )
  else:
    raise ValueError(f"Unsupported MCAP chunk compression: {compression!r}")
  if len(decoded) != uncompressed_size:
    raise ValueError(
      f"MCAP chunk size mismatch: decoded {len(decoded)}, expected {uncompressed_size}"
    )
  return decoded


def iter_policy_running_frames(path: Path, topic: str) -> Iterator[tuple[int, dict[str, Any]]]:
  """Yield ``(MCAP log time [ns], JSON frame)`` from one policy topic.

  The topic is resolved from Channel records rather than hard-coding a channel
  id, so recorder restarts and channel-id changes do not affect processing.
  """
  if not path.is_file():
    raise FileNotFoundError(path)
  channels: dict[int, str] = {}
  matched = 0
  with path.open("rb") as handle:
    if handle.read(len(_MAGIC)) != _MAGIC:
      raise ValueError(f"Not an MCAP file: {path}")
    while header := handle.read(9):
      # MCAP terminates with the eight-byte magic sequence, which is shorter
      # than a record header.
      if header == _MAGIC:
        break
      if len(header) != 9:
        raise ValueError("Truncated MCAP record header")
      opcode = header[0]
      record_size = struct.unpack("<Q", header[1:])[0]
      payload = handle.read(record_size)
      if len(payload) != record_size:
        raise ValueError("Truncated MCAP record payload")
      if opcode == _OP_CHANNEL:
        channel_id, channel_topic = _channel_topic(payload)
        channels[channel_id] = channel_topic
      elif opcode == _OP_CHUNK:
        records = _decode_chunk(payload)
        offset = 0
        while offset < len(records):
          if offset + 9 > len(records):
            raise ValueError("Truncated nested MCAP record header")
          nested_opcode = records[offset]
          nested_size = struct.unpack_from("<Q", records, offset + 1)[0]
          start = offset + 9
          end = start + nested_size
          if end > len(records):
            raise ValueError("Truncated nested MCAP record payload")
          if nested_opcode == _OP_MESSAGE:
            if nested_size < 22:
              raise ValueError("Truncated MCAP Message record")
            channel_id = struct.unpack_from("<H", records, start)[0]
            log_time_ns = struct.unpack_from("<Q", records, start + 6)[0]
            if channels.get(channel_id) == topic:
              try:
                frame = json.loads(records[start + 22:end].decode("utf-8"))
              except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Policy-running MCAP message is not JSON") from exc
              matched += 1
              yield log_time_ns, frame
          offset = end
  if not matched:
    known = ", ".join(sorted(set(channels.values()))) or "<none>"
    raise ValueError(f"MCAP topic {topic!r} was not found; channels: {known}")
