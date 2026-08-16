"""Docker / docker compose での起動構成が壊れていないことを確認する。

実際の `docker build` はCIで回さないため、コンテナが動くための条件
（依存のOSライブラリ、モデルDL、data永続化、公開ポート、起動コマンド）を検証する。
"""

from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


class TestDockerfile:
    def test_installs_runtime_deps_and_model(self):
        text = _read("Dockerfile")
        assert "python:3.11" in text
        # MediaPipe / OpenCV が必要とする共有ライブラリと、H.264再エンコード用ffmpeg
        for lib in ("libegl1", "libgl1", "libgles2", "libglib2.0-0", "ffmpeg"):
            assert lib in text, lib
        assert "requirements.txt" in text
        assert "face_landmarker.task" in text

    def test_warms_the_numba_cache_at_build_time(self):
        # pymatting の numba 関数を初回importでJITコンパイルすると 490MB 使い、glibcは
        # そのヒープをOSに返さないので 512MB ホストでは解析リクエストがOOMする。
        # ビルド時にコンパイルしてキャッシュをイメージへ焼く（起動時は読むだけ）。
        lines = [line.strip() for line in _read("Dockerfile").splitlines()]
        warm = lines.index('RUN python -c "import pymatting"')
        # 依存を入れる前には import できず、アプリのCOPYより後だとコード変更のたびに
        # このレイヤーが作り直される（＝毎デプロイでビルドが30秒延びる）
        install = next(i for i, line in enumerate(lines) if "pip install" in line)
        copy_app = next(i for i, line in enumerate(lines) if line.startswith("COPY backend"))
        assert install < warm < copy_app

    def test_pins_the_numba_target_cpu_for_build_and_runtime(self):
        # numba のキャッシュキーは対象CPU名を含むので、ビルドホストと Render の CPU が
        # 違うとキャッシュは無視され、起動時に 490MB の JIT が走って元の OOM に戻る
        # （実測: 焼いたイメージを NUMBA_CPU_NAME=generic で動かすと 24秒・453MB）。
        lines = [line.strip() for line in _read("Dockerfile").splitlines()]
        pin = lines.index("ENV NUMBA_CPU_NAME=generic \\")
        assert 'NUMBA_CPU_FEATURES=""' in lines[pin + 1]
        # ENV はイメージに残るので、ビルド時と実行時が同じターゲットを選ぶ
        assert pin < lines.index('RUN python -c "import pymatting"')

    def test_serves_the_app_on_configured_port(self):
        text = _read("Dockerfile")
        assert "backend.app:app" in text
        assert "--host 0.0.0.0" in text
        # Render injects PORT (normally 10000); local Docker keeps using 8000.
        assert "${PORT:-8000}" in text

    def test_dockerignore_excludes_local_state(self):
        ignored = _read(".dockerignore").split()
        for path in (".venv", "data", "models"):
            assert path in ignored, path


class TestCompose:
    def _service(self):
        compose = yaml.safe_load(_read("docker-compose.yml"))
        services = compose["services"]
        assert len(services) == 1
        return next(iter(services.values()))

    def test_publishes_8000_and_persists_data(self):
        service = self._service()
        assert any(str(p).startswith("8000:8000") for p in service["ports"])
        # WSL のホスト側から抽出結果・カタログを触れるようにバインドマウントする
        assert any(v.endswith(":/app/data") for v in service["volumes"])

    def test_supabase_env_is_optional(self):
        env = self._service()["environment"]
        keys = env if isinstance(env, dict) else dict(e.split("=", 1) for e in env)
        for key in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_PUBLISHABLE_KEY"):
            # 未設定でもローカル実装で動く必要があるので、既定値は空
            assert keys[key].startswith("${" + key), keys[key]
