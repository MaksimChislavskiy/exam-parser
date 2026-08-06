from __future__ import annotations

import sys

from exam_parser.entrypoint import configure_utf8_console


class ReconfigurableStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


class PlainStream:
    pass


def test_console_streams_are_reconfigured_to_utf8(monkeypatch) -> None:
    stdout = ReconfigurableStream()
    stderr = ReconfigurableStream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    configure_utf8_console()

    expected = [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert stdout.calls == expected
    assert stderr.calls == expected


def test_plain_streams_without_reconfigure_are_supported(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", PlainStream())
    monkeypatch.setattr(sys, "stderr", PlainStream())

    configure_utf8_console()
