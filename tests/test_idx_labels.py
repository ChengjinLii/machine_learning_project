"""Offline regression tests for Fashion-MNIST IDX label files."""

from __future__ import annotations

import gzip
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_course.data_utils import _read_idx_labels


class IDXLabelReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "labels.idx.gz"

    def write_label_file(self, payload: bytes) -> None:
        with gzip.open(self.path, "wb") as handle:
            handle.write(payload)

    def test_valid_labels(self) -> None:
        self.write_label_file(struct.pack(">II", 2049, 3) + bytes([0, 5, 9]))

        labels = _read_idx_labels(self.path)

        self.assertEqual(labels.dtype, "uint8")
        self.assertEqual(labels.tolist(), [0, 5, 9])

    def test_truncated_header(self) -> None:
        header = struct.pack(">II", 2049, 3)
        for length in (0, 4, 7):
            with self.subTest(header_bytes=length):
                self.write_label_file(header[:length])
                with self.assertRaisesRegex(ValueError, "Truncated IDX label header"):
                    _read_idx_labels(self.path)

    def test_invalid_magic_number(self) -> None:
        self.write_label_file(struct.pack(">II", 2051, 1) + b"\x02")

        with self.assertRaisesRegex(ValueError, "Unexpected label magic number"):
            _read_idx_labels(self.path)

    def test_empty_label_payload(self) -> None:
        """A valid zero-count IDX label file should produce an empty array."""
        self.write_label_file(struct.pack(">II", 2049, 0))

        labels = _read_idx_labels(self.path)

        self.assertEqual(labels.dtype, "uint8")
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(labels.tolist(), [])

    def test_zero_count_rejects_unexpected_payload(self) -> None:
        """A zero-count header must not silently accept trailing bytes."""
        self.write_label_file(struct.pack(">II", 2049, 0) + b"\x07")

        with self.assertRaisesRegex(ValueError, "Label count mismatch"):
            _read_idx_labels(self.path)

    def test_label_count_mismatch(self) -> None:
        header = struct.pack(">II", 2049, 2)
        for payload in (b"\x01", b"\x01\x02\x03"):
            with self.subTest(actual_count=len(payload)):
                self.write_label_file(header + payload)
                with self.assertRaisesRegex(ValueError, "Label count mismatch"):
                    _read_idx_labels(self.path)


if __name__ == "__main__":
    unittest.main()
