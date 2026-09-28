"""UCI Online Retail 的下载、解压与完整性校验。

只用标准库 ``urllib.request``：为一个 23 MB 的一次性下载引入 requests 不划算，
而且少一个依赖就少一处版本漂移。
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from experiment.data.spec import (
    UCI_URL,
    UCI_XLSX_BYTES,
    UCI_XLSX_NAME,
    UCI_XLSX_SHA256,
    UCI_ZIP_ENTRY,
    UCI_ZIP_NAME,
    USER_AGENT,
)

#: 单次下载的超时（秒）
_TIMEOUT_SECONDS = 180


def sha256_file(path: Path) -> str:
    """文件的 SHA-256 十六进制摘要，分块读取以免整文件进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_zip(url: str = UCI_URL, dest: Path | None = None) -> Path:
    """下载 zip 到 ``dest`` 并返回该路径。

    部分镜像会拒绝没有 User-Agent 的请求，故显式带上。
    """
    dest = Path(dest) if dest is not None else Path(UCI_ZIP_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return dest


def extract_entry(archive: Path, entry: str, dest: Path) -> Path:
    """把 zip 中的 ``entry`` 解到 ``dest``。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        with zf.open(entry) as source, open(dest, "wb") as target:
            shutil.copyfileobj(source, target)
    return dest


def verify_workbook_size(path: Path, expected_bytes: int = UCI_XLSX_BYTES) -> int:
    """校验工作簿字节数，不符即抛错。

    UCI 是外部来源：文件一旦被替换，清洗结果、实例构造、全部指标都会跟着变，
    而实验的可复现性依赖的正是这份固定的输入。宁可在这里硬失败。
    """
    actual = Path(path).stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"工作簿字节数不符：期望 {expected_bytes}，实际 {actual}（{path}）。"
            "上游文件可能已变更，请先核对再继续。"
        )
    return actual


def verify_workbook_digest(path: Path, expected: str = UCI_XLSX_SHA256) -> str:
    """校验工作簿 SHA-256，不符即抛错，返回实测摘要。

    字节数校验拦不住「同尺寸替换」：上游换一个恰好同样长的文件，尺寸校验会
    静默放行，而下游的一切都建立在错误的数据之上。摘要把这件事变成硬失败。
    """
    actual = sha256_file(Path(path))
    if actual != expected:
        raise ValueError(
            f"工作簿摘要不符：期望 {expected}，实际 {actual}（{path}）。"
            "上游文件可能已变更，请先核对再继续。"
        )
    return actual


def ensure_workbook(
    dest_dir: Path,
    force: bool = False,
    url: str = UCI_URL,
) -> tuple[Path, Path]:
    """确保 ``dest_dir`` 下有可用的 zip 与 xlsx，返回 ``(zip_path, xlsx_path)``。

    已存在且字节数与摘要均正确的 xlsx 会被直接复用，避免每次重跑都下载 23 MB。
    """
    dest_dir = Path(dest_dir)
    zip_path = dest_dir / UCI_ZIP_NAME
    xlsx_path = dest_dir / UCI_XLSX_NAME

    if not force and xlsx_path.is_file():
        verify_workbook_size(xlsx_path)
        verify_workbook_digest(xlsx_path)
        return zip_path, xlsx_path

    fetch_zip(url=url, dest=zip_path)
    extract_entry(zip_path, UCI_ZIP_ENTRY, xlsx_path)
    verify_workbook_size(xlsx_path)
    verify_workbook_digest(xlsx_path)
    return zip_path, xlsx_path
