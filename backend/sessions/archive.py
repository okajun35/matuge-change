"""セッションディレクトリを外部オブジェクトストレージへ退避／復元する。

別マシン・別セッションへ作業を引き継ぐための機能。ストレージ未設定でもアプリは
そのまま動く（`ArchiveUnavailable` を返すだけ）ので、ローカル利用の妨げにならない。
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from backend.sessions.errors import SessionNotFound
from backend.sessions.store import SessionStore


class ArchiveUnavailable(RuntimeError):
    pass


class ObjectArchive(Protocol):
    def upload(self, path: str, data: bytes) -> None: ...

    def download(self, path: str) -> bytes: ...

    def list(self, prefix: str) -> list[str]: ...


class SessionArchiveService:
    def __init__(self, store: SessionStore, archive: ObjectArchive | None) -> None:
        self.store = store
        self.archive = archive

    @property
    def enabled(self) -> bool:
        return self.archive is not None

    def _require_archive(self) -> ObjectArchive:
        if self.archive is None:
            raise ArchiveUnavailable("session archive needs Supabase Storage to be configured")
        return self.archive

    def export(self, session_id: str) -> dict[str, Any]:
        archive = self._require_archive()
        directory = self.store.require(session_id)

        names = sorted(n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n)))
        for name in names:
            with open(os.path.join(directory, name), "rb") as f:
                archive.upload(f"{session_id}/{name}", f.read())
        return {"session_id": session_id, "files": names}

    def restore(self, session_id: str) -> dict[str, Any]:
        archive = self._require_archive()

        paths = archive.list(session_id)
        if not paths:
            raise SessionNotFound(session_id)

        directory = os.path.join(self.store.root, session_id)
        os.makedirs(directory, exist_ok=True)

        names = []
        for path in paths:
            name = path.split("/", 1)[1]
            with open(os.path.join(directory, name), "wb") as f:
                f.write(archive.download(path))
            names.append(name)
        return {"session_id": session_id, "files": sorted(names)}
