from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from derail.world.facts import Fact

FILE_TABLE = "files.documents"
FILE_COLUMNS = ("path", "name", "mime", "size", "sha256", "content", "lines")
INVENTORY_VERSION = "file-inventory/1.0"
_MIME = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".fods": "application/vnd.oasis.opendocument.spreadsheet-flat-xml",
    ".fodt": "application/vnd.oasis.opendocument.text-flat-xml",
    ".fodp": "application/vnd.oasis.opendocument.presentation-flat-xml",
    ".odt": "application/vnd.oasis.opendocument.text",
}
_XML_TEXT = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
_XML_PARA = re.compile(r"</w:p>")
_SHEET_CELL = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>', re.DOTALL)
_SHARED = re.compile(r"<si>.*?</si>", re.DOTALL)
_T = re.compile(r"<t[^>]*>([^<]*)</t>")


def file_entity(relative_path: str) -> str:
    return "file:%s" % relative_path.lstrip("/")


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", "replace")
    paragraphs = [" ".join(_XML_TEXT.findall(p)) for p in _XML_PARA.split(xml)]
    return "\n".join(p for p in paragraphs if p.strip())


def _xlsx_rows(data: bytes) -> List[List[str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in _SHARED.findall(
                archive.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            ):
                shared.append("".join(_T.findall(item)))
        sheets = sorted(
            n
            for n in archive.namelist()
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        rows: Dict[int, Dict[str, str]] = {}
        for sheet in sheets[:1]:
            xml = archive.read(sheet).decode("utf-8", "replace")
            for col, row, attrs, value in _SHEET_CELL.findall(xml):
                text = (
                    shared[int(value)]
                    if 't="s"' in attrs and value.isdigit() and int(value) < len(shared)
                    else value
                )
                rows.setdefault(int(row), {})[col] = text
    out = []
    for index in sorted(rows):
        cells = rows[index]
        out.append([cells[c] for c in sorted(cells, key=lambda k: (len(k), k))])
    return out


_ODF_BLOCK = re.compile(r"</(?:text:p|text:h|table:table-row|draw:page)>")
_TAG = re.compile(r"<[^>]+>")
_ODF_CELL_SEP = re.compile(r"</table:table-cell>")


def _odf_xml_text(xml: str) -> str:
    xml = _ODF_CELL_SEP.sub("\t", xml)
    xml = (
        xml.replace("<text:tab/>", "\t")
        .replace("<text:line-break/>", "\n")
        .replace("<text:s/>", " ")
    )
    lines = [_TAG.sub("", block).strip() for block in _ODF_BLOCK.split(xml)]
    body = "\n".join(line for line in lines if line.strip())
    return (
        body.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


_PDF_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_TJ = re.compile(rb"\((.*?)(?<!\\)\)\s*Tj")
_PDF_TJ_ARRAY = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
_PDF_ARRAY_STR = re.compile(rb"\((.*?)(?<!\\)\)")


def _pdf_text(data: bytes) -> Optional[str]:
    import zlib

    lines: List[str] = []
    for stream in _PDF_STREAM.findall(data):
        content = stream
        if b"Tj" not in content and b"TJ" not in content:
            try:
                content = zlib.decompress(stream)
            except zlib.error:
                continue
        for block in re.split(rb"\bET\b", content):
            if b"BT" not in block:
                continue
            parts = [m.group(1) for m in _PDF_TJ.finditer(block)]
            for array in _PDF_TJ_ARRAY.findall(block):
                parts.append(b"".join(_PDF_ARRAY_STR.findall(array)))
            text = (
                b" ".join(parts)
                .replace(b"\\(", b"(")
                .replace(b"\\)", b")")
                .decode("latin-1", "replace")
                .strip()
            )
            if text:
                lines.append(text)
    return "\n".join(lines) if lines else None


def _plain_text(data: bytes) -> Optional[str]:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def extract_text(path: Union[str, Path]) -> Optional[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix in (".txt", ".md", ".csv", ".json", ""):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    if suffix == ".docx":
        try:
            return _docx_text(data)
        except (zipfile.BadZipFile, KeyError):
            return _plain_text(data)
    if suffix == ".xlsx":
        try:
            return "\n".join("\t".join(row) for row in _xlsx_rows(data))
        except (zipfile.BadZipFile, KeyError):
            return _plain_text(data)
    if suffix in (".fods", ".fodt", ".fodp"):
        return _odf_xml_text(data.decode("utf-8", "replace"))
    if suffix in (".odt", ".ods", ".odp"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return _odf_xml_text(archive.read("content.xml").decode("utf-8", "replace"))
        except (zipfile.BadZipFile, KeyError):
            return _plain_text(data)
    if suffix == ".pdf":
        return _pdf_text(data)
    return None


@dataclass
class FileEntry:
    path: str
    size: int
    sha256: str
    mime: str
    content: Optional[str]
    lines: Optional[int]

    @property
    def entity(self) -> str:
        return file_entity(self.path)

    def row(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": Path(self.path).name,
            "mime": self.mime,
            "size": self.size,
            "sha256": self.sha256,
            "content": self.content,
            "lines": self.lines,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "mime": self.mime,
            "content": self.content,
            "lines": self.lines,
        }


@dataclass
class FileInventory:
    home_prefix: str
    entries: Dict[str, FileEntry] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        root: Union[str, Path],
        home_prefix: str = "/home/user",
        include: Iterable[str] = ("Documents", "Desktop", "Downloads"),
    ) -> "FileInventory":
        root = Path(root)
        inventory = cls(home_prefix=home_prefix)
        for top in include:
            base = root / top
            if not base.is_dir():
                continue
            for path in sorted(p for p in base.rglob("*") if p.is_file()):
                relative = str(path.relative_to(root))
                data = path.read_bytes()
                text = extract_text(path)
                inventory.entries[relative] = FileEntry(
                    path=relative,
                    size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    mime=_MIME.get(path.suffix.lower(), "application/octet-stream"),
                    content=text,
                    lines=(text.count("\n") + 1) if text else None,
                )
        return inventory

    def get(self, key: str) -> Optional[FileEntry]:
        path = key[len("file:") :] if key.startswith("file:") else key
        if path.startswith(self.home_prefix + "/"):
            path = path[len(self.home_prefix) + 1 :]
        return self.entries.get(path.lstrip("/"))

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": INVENTORY_VERSION,
            "home_prefix": self.home_prefix,
            "files": [e.to_dict() for e in self.entries.values()],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FileInventory":
        inventory = cls(home_prefix=str(record.get("home_prefix", "/home/user")))
        for item in record.get("files", ()):
            inventory.entries[item["path"]] = FileEntry(
                item["path"],
                int(item["size"]),
                item["sha256"],
                item["mime"],
                item.get("content"),
                item.get("lines"),
            )
        return inventory

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(self.to_record(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FileInventory":
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary_lines(self, max_chars: int = 160) -> List[str]:
        out = []
        for entry in self.entries.values():
            excerpt = (entry.content or "").strip().replace("\n", " / ")[:max_chars]
            out.append(
                "- file:%s (%s, %d bytes)%s"
                % (
                    entry.path,
                    entry.mime.split("/")[-1],
                    entry.size,
                    (": " + excerpt) if excerpt else "",
                )
            )
        return out


def file_facts(entry: FileEntry, columns: Iterable[str] = FILE_COLUMNS) -> List[Fact]:
    row = entry.row()
    return [
        Fact(FILE_TABLE, column, entry.entity, row[column]) for column in columns if column in row
    ]


def csv_rows(text: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))
