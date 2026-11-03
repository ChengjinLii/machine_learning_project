"""Regression tests for the Fashion-MNIST IDX image reader (no downloads)."""

from __future__ import annotations

import gzip
import struct
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_course.data_utils import _read_idx_images


class IDXImageReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "images.idx.gz"

    def write_image_file(self, payload: bytes) -> None:
        with gzip.open(self.path, "wb") as handle:
            handle.write(payload)

    def test_valid_images(self) -> None:
        header = struct.pack(">IIII", 2051, 2, 2, 2)
        self.write_image_file(header + bytes(range(8)))

        images = _read_idx_images(self.path)

        self.assertEqual(images.shape, (2, 4))
        self.assertEqual(images[0].tolist(), [0, 1, 2, 3])

    def test_truncated_header(self) -> None:
        self.write_image_file(struct.pack(">II", 2051, 2))

        with self.assertRaisesRegex(ValueError, "Truncated IDX image header"):
            _read_idx_images(self.path)

    def test_invalid_magic_number(self) -> None:
        header = struct.pack(">IIII", 9999, 1, 2, 2)
        self.write_image_file(header + bytes(range(4)))

        with self.assertRaisesRegex(ValueError, "Unexpected image magic number"):
            _read_idx_images(self.path)

    def test_payload_length_mismatch(self) -> None:
        header = struct.pack(">IIII", 2051, 2, 2, 2)
        for pixels in (7, 9):
            with self.subTest(pixel_count=pixels):
                self.write_image_file(header + bytes(range(pixels)))
                with self.assertRaisesRegex(ValueError, "Image payload size mismatch"):
                    _read_idx_images(self.path)


if __name__ == "__main__":
    unittest.main()
