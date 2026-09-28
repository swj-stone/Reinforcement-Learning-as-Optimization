"""下载与解压的离线测试。

网络部分不在这里测：CI 与离线环境都必须能跑绿。真实下载由
tests/test_collect_integration.py 负责，那个文件在缺文件时自动跳过。
"""

import zipfile

import pytest

from experiment.data.collect.download import (
    extract_entry,
    sha256_file,
    verify_workbook_digest,
    verify_workbook_size,
)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello knapsack")
    assert sha256_file(path) == hashlib.sha256(b"hello knapsack").hexdigest()


def test_extract_entry_writes_the_named_member(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Online Retail.xlsx", b"xlsx-bytes")
        zf.writestr("readme.txt", b"ignore me")

    dest = tmp_path / "out.xlsx"
    extract_entry(archive, "Online Retail.xlsx", dest)
    assert dest.read_bytes() == b"xlsx-bytes"


def test_extract_entry_missing_member_raises(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("other.xlsx", b"nope")
    with pytest.raises(KeyError):
        extract_entry(archive, "Online Retail.xlsx", tmp_path / "out.xlsx")


def test_verify_workbook_size_accepts_a_matching_file(tmp_path):
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    assert verify_workbook_size(path, expected_bytes=5) == 5


def test_verify_workbook_size_rejects_a_changed_file(tmp_path):
    """UCI 若替换了源文件，必须报错停下，而不是静默换掉整个实验的数据基础。"""
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="字节数"):
        verify_workbook_size(path, expected_bytes=6)


def test_verify_workbook_digest_accepts_a_matching_file(tmp_path):
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    expected = sha256_file(path)
    assert verify_workbook_digest(path, expected=expected) == expected


def test_verify_workbook_digest_rejects_a_same_size_swap(tmp_path):
    """字节数校验拦不住同尺寸替换：内容变了，尺寸照旧。摘要必须单独拦下它。"""
    path = tmp_path / "w.xlsx"
    path.write_bytes(b"12345")
    before = verify_workbook_digest(path, expected=sha256_file(path))

    path.write_bytes(b"12346")  # 同样 5 字节，内容不同
    assert verify_workbook_size(path, expected_bytes=5) == 5  # 尺寸校验静默放行
    with pytest.raises(ValueError, match="摘要"):
        verify_workbook_digest(path, expected=before)
