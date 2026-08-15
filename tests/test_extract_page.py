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


class TestRoiModeToggle:
    """正面（自動顔検出）と横顔・目アップ（手動ROI）をUIで切り替えられること。"""

    def test_mode_selector_exists(self):
        page = _page()
        assert "roiMode" in page
        assert "manual" in page and "auto" in page

    def test_manual_mode_sends_the_drawn_rect(self):
        page = _page()
        assert "roi_rect" in page
        assert "roiA" in page and "roiB" in page

    def test_rect_is_drawn_on_the_local_preview(self):
        page = _page()
        # 解析前のローカルプレビュー上でドラッグしてROIを決める
        assert "roiOverlay" in page

    def test_auto_mode_defaults_and_falls_back_to_manual_when_no_face_is_found(self):
        page = _page()
        # デフォルトは自動（selectの先頭がauto）
        assert page.index('value="auto"') < page.index('value="manual"')
        # 顔検出できなかったときはエラーで止めず、手動ROIへ切り替えて案内する
        assert "fallbackToManualRoi" in page
        assert "no face" in page

    def test_manual_roi_a_and_b_controls_are_present(self):
        page = _page()
        for marker in ("roiBtnA", "roiBtnB", "roiClear", "roiShow"):
            assert marker in page
        assert "ROI-A 指定（装着画像）" in page
        assert "ROI-B 指定（加工画像）" in page

    def test_manual_roi_overlay_colors_and_recompose_dest_rect(self):
        page = _page()
        assert "#00e5ff" in page and "#ff9f1c" in page
        assert "roiOverlayA" in page and "roiOverlayB" in page
        assert "dest_rect" in page
        assert "加工画像で ROI-B を指定してください" in page

    def test_arming_roi_refreshes_button_and_hint_immediately(self):
        page = _page()
        assert "showLayer(target);\n  updateRoiHint();" in page
        assert "state.activeRoi = null;\n  updateRoiHint();" in page
        assert "roiModeSelect.onchange" in page and "updateRoiHint();" in page

    def test_show_layer_keeps_layer_selector_in_sync(self):
        page = _page()
        assert "document.getElementById('layerSelect').value = name;" in page


class TestRestoredStrokesArePainted:
    def test_strokes_are_replayed_after_the_layer_image_finished_loading(self):
        page = _page()
        # レイヤー画像の onload が paintCanvas をリサイズ（=内容クリア）するため、
        # 表示完了を待たずにストロークを再生すると復元直後の筆跡が消える
        assert "return new Promise" in page
        assert "await setLayerOptions" in page
        assert page.index("await setLayerOptions") < page.index("await loadStrokes")


class TestSessionArchiveUi:
    def test_archive_buttons_call_the_archive_endpoints(self):
        page = _page()
        assert "btnArchive" in page and "btnArchiveRestore" in page
        assert "/archive'" in page or "/archive`" in page
        assert "/archive/restore" in page


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
