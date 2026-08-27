from __future__ import annotations

import gzip

from daralm.data.commoncrawl import (
    _decompressed_chunks,
    _record_language,
    extract_warc_html,
    html_to_text,
    iter_wet_records,
    remove_repeated_boilerplate,
)


def _wet_record(text: str, **headers: str) -> bytes:
    payload = text.encode()
    fields = {
        "WARC-Type": "conversion",
        "WARC-Target-URI": "https://example.test/page",
        "Content-Length": str(len(payload)),
        **headers,
    }
    header = "WARC/1.0\r\n" + "".join(f"{key}: {value}\r\n" for key, value in fields.items())
    return header.encode() + b"\r\n" + payload + b"\r\n\r\n"


def test_iter_wet_records_handles_split_chunks():
    raw = _wet_record("hello world", **{"WARC-Identified-Content-Language": "eng"})
    records = list(iter_wet_records([raw[:7], raw[7:31], raw[31:]]))
    assert len(records) == 1
    headers, payload = records[0]
    assert headers["warc-target-uri"] == "https://example.test/page"
    assert payload == b"hello world"


def test_iter_wet_records_handles_multiple_records():
    raw = _wet_record("first") + _wet_record("second")
    assert [payload for _, payload in iter_wet_records([raw])] == [b"first", b"second"]


def test_record_language_rejects_metadata_that_conflicts_with_script():
    headers = {"warc-identified-content-language": "khm"}
    assert _record_language(headers, "Latin fallback text") == "en"


def test_record_language_rejects_false_english_metadata():
    headers = {"warc-identified-content-language": "eng"}
    assert _record_language(headers, "俄罗斯中文网页") is None


def test_record_language_falls_back_to_script_detection():
    assert _record_language({}, "កម្ពុជាជាប្រទេសមួយ") == "km"
    assert _record_language({}, "Cambodia is a country") == "en"


def test_record_language_rejects_unrelated_scripts():
    assert _record_language({"warc-identified-content-language": "rus"}, "Россия") is None


def test_decompressed_chunks_reads_concatenated_gzip_members():
    compressed = gzip.compress(b"first") + gzip.compress(b"second")
    split_chunks = [compressed[:9], compressed[9:25], compressed[25:]]
    assert b"".join(_decompressed_chunks(split_chunks)) == b"firstsecond"


def test_html_to_text_omits_scripts_and_styles():
    html = "<style>bad</style><h1>កម្ពុជា</h1><script>worse</script><p>អត្ថបទ</p>"
    assert html_to_text(html) == "កម្ពុជា\nអត្ថបទ"


def test_extract_warc_html():
    html = "<html><body><h1>កម្ពុជា</h1><p>អត្ថបទសាកល្បង</p></body></html>"
    http = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + html.encode()
    warc = b"WARC/1.0\r\nWARC-Type: response\r\n\r\n" + http
    assert extract_warc_html(gzip.compress(warc)) == "កម្ពុជា\nអត្ថបទសាកល្បង"


def test_remove_repeated_boilerplate_per_domain():
    records = [
        {"url": f"https://example.test/{i}", "text": f"Menu\nShared footer\nUnique article {i}"}
        for i in range(3)
    ]
    cleaned, removed = remove_repeated_boilerplate(records)
    assert [record["text"] for record in cleaned] == [
        "Unique article 0",
        "Unique article 1",
        "Unique article 2",
    ]
    assert removed == 6
