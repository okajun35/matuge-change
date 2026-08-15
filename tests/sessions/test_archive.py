"""セッションディレクトリを外部ストレージへ退避／復元するユースケース。"""

from __future__ import annotations

import os

import pytest

from backend.sessions.archive import ArchiveUnavailable, SessionArchiveService
from backend.sessions.store import SessionStore


class FakeArchive:
    """Supabase Storage の代わりに使うインメモリ実装。"""

    def __init__(self, *, available: bool = True) -> None:
        self.objects: dict[str, bytes] = {}
        self.available = available

    def upload(self, path: str, data: bytes) -> None:
        self.objects[path] = data

    def download(self, path: str) -> bytes:
        return self.objects[path]

    def list(self, prefix: str) -> list[str]:
        return sorted(p for p in self.objects if p.startswith(f"{prefix}/"))


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(str(tmp_path / "data"))


def _seed(store: SessionStore) -> str:
    session_id = store.create()
    with open(store.path(session_id, "roi_a.png"), "wb") as f:
        f.write(b"png-bytes")
    with open(store.path(session_id, "meta.json"), "w") as f:
        f.write('{"width": 4}')
    return session_id


class TestExport:
    def test_uploads_every_session_file_under_the_session_prefix(self, store):
        archive = FakeArchive()
        session_id = _seed(store)

        result = SessionArchiveService(store, archive).export(session_id)

        assert set(result["files"]) == {"roi_a.png", "meta.json"}
        assert archive.objects[f"{session_id}/roi_a.png"] == b"png-bytes"
        assert result["session_id"] == session_id

    def test_unknown_session_is_rejected(self, store):
        from backend.sessions.errors import SessionNotFound

        with pytest.raises(SessionNotFound):
            SessionArchiveService(store, FakeArchive()).export("nope")

    def test_without_an_archive_backend_it_reports_unavailable(self, store):
        session_id = _seed(store)
        with pytest.raises(ArchiveUnavailable):
            SessionArchiveService(store, None).export(session_id)


class TestImport:
    def test_downloads_into_a_local_session_dir(self, store, tmp_path):
        archive = FakeArchive()
        session_id = _seed(store)
        SessionArchiveService(store, archive).export(session_id)

        # 別マシン相当の空ストアへ復元する
        fresh = SessionStore(str(tmp_path / "other"))
        result = SessionArchiveService(fresh, archive).restore(session_id)

        assert set(result["files"]) == {"roi_a.png", "meta.json"}
        assert fresh.exists(session_id)
        with open(fresh.path(session_id, "roi_a.png"), "rb") as f:
            assert f.read() == b"png-bytes"

    def test_restoring_twice_is_idempotent(self, store, tmp_path):
        archive = FakeArchive()
        session_id = _seed(store)
        SessionArchiveService(store, archive).export(session_id)

        fresh = SessionStore(str(tmp_path / "other"))
        service = SessionArchiveService(fresh, archive)
        service.restore(session_id)
        service.restore(session_id)

        assert os.path.exists(fresh.path(session_id, "meta.json"))

    def test_missing_archive_entry_is_reported_as_not_found(self, store):
        from backend.sessions.errors import SessionNotFound

        with pytest.raises(SessionNotFound):
            SessionArchiveService(store, FakeArchive()).restore("ghost")
