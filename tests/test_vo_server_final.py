"""Final-review regression: `astrosearch vo ... --format csv` to stdout keeps the CSV's CRLF line
endings (a text-mode stdout on Windows turned them into CR CR LF: a blank line after every row)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import vo_server

ROOT = Path(vo_server.__file__).resolve().parent


def test_write_bytes_bypasses_newline_translation() -> None:
    body = b"name,wavelength\r\ngaia_dr3,optical\r\n"
    # A text stream that translates every "\n" it is given into "\r\n" (Windows' stdout does).
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="\r\n")
    stream.write("x")  # pending text is flushed first, in order
    vo_server.write_bytes(stream, body)
    assert stream.buffer.getvalue() == b"x" + body
    plain = io.StringIO()  # no binary buffer (pytest's capsys): decoded text
    vo_server.write_bytes(plain, body)
    assert plain.getvalue() == body.decode()


def test_vo_adql_csv_on_stdout_has_one_crlf_per_row() -> None:
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "main.py", "vo", "adql",
                           "SELECT TOP 2 name, wavelength FROM astrosearch.catalogs", "--format", "csv"],
                          cwd=ROOT, capture_output=True, env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    out = done.stdout
    assert b"\r\r\n" not in out
    lines = out.split(b"\r\n")
    assert lines[0] == b"name,wavelength" and lines[-1] == b"" and len(lines) == 4, out
