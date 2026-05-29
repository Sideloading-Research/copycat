"""Standalone replacements for langchain_community.document_loaders.

langchain-community was sunset on 2026-05-26.
These thin wrappers keep DirectoryLoader + TextLoader working
without depending on an unmaintained package.
"""

import os
from pathlib import Path
from typing import Iterator, List, Optional
from langchain_core.documents import Document


class TextLoader:
    """Loads a single text file as a Document."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        return [Document(page_content=text)]

    def lazy_load(self) -> Iterator[Document]:
        yield from self.load()


class DirectoryLoader:
    """Loads documents from a directory with glob and error handling.

    API-compatible with langchain_community DirectoryLoader.
    """

    def __init__(
        self,
        path: str,
        glob: str = "**/[!.]*",
        silent_errors: bool = False,
        loader_cls=None,
        loader_kwargs: Optional[dict] = None,
        use_multithreading: bool = False,
        show_progress: bool = False,
        **kwargs,
    ):
        self.path = path
        self.glob = glob
        self.silent_errors = silent_errors
        self.loader_cls = loader_cls or TextLoader
        self.loader_kwargs = loader_kwargs or {}
        self.multithread = use_multithreading
        self.show_progress = show_progress

    def load(self) -> List[Document]:
        p = Path(self.path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")

        pattern = self.glob
        if "**" in pattern:
            files = list(p.rglob(pattern.lstrip("/")))
        else:
            files = list(p.glob(pattern))

        docs: List[Document] = []
        for fpath in sorted(files):
            if fpath.is_dir():
                continue
            try:
                loader = self.loader_cls(str(fpath), **self.loader_kwargs)
                docs.extend(loader.load())
            except Exception as exc:
                if not self.silent_errors:
                    raise
        return docs
