"""静止画モードのビューア要件を静的に検証する。

JSのテストランナーは持たないため、ページが備えるべき仕組み
（アップロード直後のローカルプレビュー、ズーム＋スクロール表示）を検証する。
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _page() -> str:
    with open(os.path.join(ROOT, "frontend", "extract.html"), encoding="utf-8") as f:
        return f.read()


class TestLocalPreview:
    def test_file_inputs_trigger_preview_before_analysis(self):
        page = _page()
        # 解析（セッション作成）前に選んだファイルをそのまま表示できること
        assert "createObjectURL" in page
        for input_id in ("fileWith", "fileWithout", "fileEdited"):
            assert f"'{input_id}'" in page
        assert "local_with" in page and "local_without" in page and "local_edited" in page

    def test_local_layers_do_not_require_a_session(self):
        page = _page()
        assert "isLocalLayer" in page


class TestZoomableViewer:
    def test_stage_is_a_scrollable_viewport(self):
        page = _page()
        assert "canvasWrap" in page
        assert "overflow: auto" in page

    def test_zoom_controls_exist(self):
        page = _page()
        for control in ("btnZoomIn", "btnZoomOut", "btnZoomFit", "btnZoomReset", "zoomLabel"):
            assert control in page

    def test_brush_coordinates_are_divided_by_zoom(self):
        page = _page()
        # ズーム表示中もブラシ座標が画像座標にマップされること
        assert "/ state.zoom" in page
