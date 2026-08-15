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

    def test_manual_fit_controls_exist(self):
        page = _page()
        for marker in ("btnFit", "fitCanvas", "fitScale", "fitAngle", "左右反転", "fitReset"):
            assert marker in page

    def test_fit_canvas_is_overlay_inside_canvas_wrap(self):
        page = _page()
        assert '<canvas id="fitCanvas"></canvas>' in page
        assert page.index('<div id="canvasWrap">') < page.index('<canvas id="fitCanvas"></canvas>')
        assert "fitCanvas.width = paintCanvas.width" not in page

    def test_recompose_sends_angle_and_flip(self):
        page = _page()
        assert "fd.append('angle'" in page
        assert "fd.append('flip'" in page

    def test_fit_keyboard_shortcuts_skip_focused_inputs(self):
        page = _page()
        assert "/input|textarea|select/i.test(e.target.tagName)" in page
        assert "fitMode" in page

    def test_fit_rotation_handle_hit_test_matches_drawn_position(self):
        page = _page()
        assert "local[1] + geometry.h / 2 + 30 / state.zoom" in page

    def test_fit_center_is_initialized_when_fit_starts_and_roi_changes(self):
        page = _page()
        assert "function initializeFitCenter(force = false)" in page
        assert "initializeFitCenter();" in page

    def test_fit_result_resets_scale_and_center(self):
        page = _page()
        assert "state.fitScale = 100;" in page
        assert "state.fitCenter = [(state.roiB[0] + state.roiB[2]) / 2" in page

    def test_fit_exit_keeps_available_controls_visible(self):
        page = _page()
        assert "state.fitMode = false;\n    updateFitAvailability();" in page
        assert page.count("state.fitMode = false;\n    updateFitAvailability();") >= 2

    def test_fit_drag_starts_only_inside_transformed_frame(self):
        page = _page()
        assert "const inside = Math.abs(local[0]) <= geometry.w / 2" in page
        assert "if (!inside && !nearCorner && !rotationHandle) return;" in page

    def test_resume_resets_fit_scale_input(self):
        page = _page()
        assert "state.fitScale = 100;" in page
        assert "document.getElementById('fitScale').value = state.fitScale;" in page

    def test_fit_availability_updates_before_layer_image_load(self):
        page = _page()
        assert "state.layer = name;\n  updateFitAvailability();" in page

    def test_fit_availability_updates_when_layer_image_fails(self):
        page = _page()
        assert "img.onerror = () => {\n      updateFitAvailability();" in page

    def test_fit_controls_stay_visible_and_are_disabled_when_unavailable(self):
        page = _page()
        # 条件付きで出たり消えたりすると見つけられないので、常時表示して非活性で示す
        assert '<span id="fitControls" style="display:none">' not in page
        assert "controls.style.display" not in page
        assert ".disabled = !available" in page

    def test_fit_controls_tooltip_explains_how_to_enable(self):
        page = _page()
        # 使えない理由（未解析・自動モード・別レイヤー表示中）をツールチップで案内する
        assert "controls.title" in page
        for reason in ("解析開始", "手動ROI", "加工画像"):
            assert reason in page


class TestBrushVisibilityToggle:
    """描いたブラシがレイヤー確認の邪魔になるので、表示だけをON/OFFできること。"""

    def test_toggle_exists_and_controls_the_paint_canvas(self):
        page = _page()
        assert 'id="brushShow"' in page
        assert "ブラシ表示" in page
        # 表示専用レイヤーでは従来どおり非表示のまま（チェックONでも出さない）
        assert "isViewOnlyLayer(state.layer) || !brushShow.checked" in page

    def test_hiding_strokes_does_not_affect_matting_constraints(self):
        page = _page()
        # 表示OFFは見た目だけ。制約PNGは paintCanvas の中身から作られ続ける
        assert "paintCanvas.toDataURL" in page

    def test_selecting_a_brush_tool_reenables_visibility(self):
        page = _page()
        # 非表示のまま描けると混乱するので、ブラシを選んだら表示に戻す
        assert "brushShow.checked = true" in page


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


class TestGroupedControls:
    """コントロールがワークフロー手順ごとにグループ化されていること。"""

    def test_controls_are_grouped_by_workflow_step(self):
        page = _page()
        legends = (
            "① 入力と解析",
            "② ブラシ補正とMatting",
            "③ AI加工画像へ再合成",
            "④ 商品登録",
            "セッション",
        )
        for legend in legends:
            assert f"<legend>{legend}</legend>" in page
        # 手順の順に並んでいること
        positions = [page.index(f"<legend>{legend}</legend>") for legend in legends]
        assert positions == sorted(positions)

    def test_view_toolbar_sits_directly_above_the_stage(self):
        page = _page()
        # 表示レイヤー・ズーム・保存は常時使うので、画像ビューアの直前に独立したバーを置く
        view_bar = page.index('id="viewBar"')
        stage = page.index('<div id="stage">')
        assert view_bar < stage
        body = page.index("<body>")
        html = page[body:stage]
        for control in ("layerSelect", "btnZoomIn", "btnDownload"):
            assert control in html


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

    def test_zoom_persists_when_switching_between_same_size_layers(self):
        page = _page()
        # 解像度が同じレイヤー間の切替では拡大縮小率を維持し、
        # 解像度が変わったとき（ROIレイヤー ↔ 元画像）だけ自動フィット/等倍にする
        assert (
            "const sizeChanged = baseCanvas.width !== img.width || baseCanvas.height !== img.height;" in page
        )
        # サイズ比較はキャンバスを新しい画像サイズに書き換える前に行うこと
        assert page.index("const sizeChanged") < page.index("baseCanvas.width = img.width")
        assert "if (sizeChanged)" in page
        # サイズが変わらないときもスクロール範囲の再計算は行う
        assert "else applyZoom();" in page
