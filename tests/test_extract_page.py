"""静止画モードのビューア要件を静的に検証する。

JSのテストランナーは持たないため、ページが備えるべき仕組み
（アップロード直後のローカルプレビュー、ズーム＋スクロール表示）を検証する。
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _page() -> str:
    with open(os.path.join(ROOT, "frontend", "extract.html"), encoding="utf-8") as f:
        return f.read()


class TestSimpleMode:
    """初見利用者向けの一括処理が、詳細調整より先に提示されること。"""

    def test_simple_mode_is_the_default_and_advanced_controls_are_collapsed(self):
        page = _page()
        assert 'id="simpleMode"' in page
        assert 'id="advancedMode" hidden' in page
        assert page.index('id="simpleMode"') < page.index('id="advancedMode"')

    def test_two_required_image_drop_zones_use_the_existing_file_inputs(self):
        page = _page()
        for zone, input_id, label in (
            ("dropWith", "fileWith", "装着画像"),
            ("dropEdited", "fileEdited", "AI加工済み画像"),
        ):
            assert f'id="{zone}"' in page
            assert f'for="{input_id}"' in page
            assert f'id="{input_id}"' in page
            assert label in page
        assert 'id="simpleRun" disabled' in page

    def test_drop_zones_accept_dropped_image_files_and_reject_other_files(self):
        page = _page()
        assert "dataTransfer.files" in page
        assert "file.type.startsWith('image/')" in page
        assert "このファイルは画像ではありません" in page
        assert "dragover" in page and "drop" in page

    def test_processing_modal_uses_plain_language_steps(self):
        page = _page()
        assert 'id="processModal"' in page
        for step in (
            "画像を確認しています",
            "目元を検出しています",
            "まつ毛を抽出しています",
            "加工画像に合成しています",
            "仕上がりを準備しています",
        ):
            assert step in page
        assert "showModal()" in page

    def test_simple_flow_reuses_the_three_existing_processing_steps_in_order(self):
        page = _page()
        start = page.index("async function runSimpleFlow()")
        end = page.index("\n}", start)
        flow = page[start:end]
        calls = [flow.index(name) for name in ("createSession()", "runMatting()", "recompose()")]
        assert calls == sorted(calls)

    def test_success_actions_compare_download_and_open_the_same_advanced_session(self):
        page = _page()
        for marker in ("simpleResult", "simpleCompare", "simpleDownload", "openAdvancedMode"):
            assert marker in page
        assert "composite_on_edited" in page
        assert "source_edited" in page

    def test_failure_offers_detailed_adjustment_without_discarding_the_session(self):
        page = _page()
        assert 'id="processOpenAdvanced"' in page
        assert "詳細調整を開く" in page
        assert "state.session = null" not in page

    def test_completed_result_is_at_least_100_percent_and_centered_on_the_product(self):
        page = _page()
        assert "function focusSimpleResult(focusRect)" in page
        assert "Math.max(1," in page
        assert "stage.scrollLeft" in page and "stage.scrollTop" in page
        start = page.index("async function runSimpleFlow()")
        end = page.index("\n}", start)
        flow = page[start:end]
        assert "const recomposeResult = await recompose();" in flow
        assert "focusSimpleResult(recomposeResult.focus_rect);" in flow

    def test_simple_layers_prioritize_the_three_user_images_then_extraction_outputs(self):
        page = _page()
        primary = (
            ("simpleLayerOriginal", "装着画像（元）"),
            ("simpleLayerEdited", "AI加工済み画像"),
            ("simpleLayerResult", "合成結果"),
        )
        positions = []
        for marker, label in primary:
            assert f'id="{marker}"' in page
            assert label in page
            positions.append(page.index(f'id="{marker}"'))
        assert positions == sorted(positions)
        assert 'id="simpleExtractionLayers"' in page
        assert "まつ毛の切り抜き" in page
        assert "透過マスク（Alpha）" in page

    def test_comparison_places_result_left_and_original_right(self):
        page = _page()
        assert 'id="compareMode" hidden' in page
        left = page.index('id="compareResultCanvas"')
        right = page.index('id="compareOriginalCanvas"')
        assert left < right
        assert "左：まつ毛装着後のAI画像" in page
        assert "右：オリジナル装着画像" in page

    def test_comparison_starts_on_lashes_and_minus_moves_toward_the_whole_image(self):
        page = _page()
        assert "compareZoomLevel: 1" in page
        assert "state.compareZoomLevel - COMPARE_ZOOM_STEP" in page
        assert "state.compareZoomLevel = 0" in page
        assert "state.compareZoomLevel = 1" in page
        assert "sourceFocusRect" in page and "simpleFocusRect" in page
        assert "drawComparison()" in page

    def test_comparison_can_stack_at_the_device_width_on_mobile(self):
        page = _page()
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
        assert "@media (max-width: 700px)" in page
        assert ".compare-grid { grid-template-columns: 1fr; }" in page

    def test_success_opens_comparison_before_the_layer_result_view(self):
        page = _page()
        start = page.index("async function runSimpleFlow()")
        end = page.index("\n}", start)
        flow = page[start:end]
        assert "await openComparison();" in flow
        assert flow.index("await recompose();") < flow.index("await openComparison();")
        assert "simpleResult.hidden = false;" not in flow

    def test_comparison_can_download_the_full_resolution_ai_lash_result(self):
        page = _page()
        assert 'id="compareDownload"' in page
        assert 'download="matuge-ai-result.png"' in page
        assert ">結果を保存</a>" in page
        assert "compareDownload.href = `/api/image/${state.session}/composite_on_edited`;" in page


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
        # 以前の display:none での出し入れに戻っていないこと
        assert "controls.style.display = available" not in page
        assert ".disabled = !available" in page
        # :has() 非対応ブラウザでも非活性の見た目が変わるよう、JS側でも opacity を設定する
        assert "controls.style.opacity" in page

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


class TestValidHtmlStructure:
    def test_no_buttons_nested_inside_links(self):
        page = _page()
        # <a> の中に <button> を入れるのは無効なHTML（インタラクティブ要素のネスト）
        assert not re.search(r"<a[^>]*>\s*<button", page)


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


class TestLayerSelector:
    """表示レイヤーの選択肢が、何の画像かとどの手順で作られるかを説明していること。"""

    def test_initial_option_is_a_placeholder_not_a_real_layer(self):
        page = _page()
        # roi_a は解析後にしか存在しないので、初期状態で選べる項目として置かない
        assert '<option value="roi_a">' not in page
        assert re.search(r'<option value="" disabled selected>[^<]+</option>', page)

    def test_choosing_a_layer_before_analysis_explains_why_nothing_appears(self):
        page = _page()
        # 無言で return すると「機能していない」と誤解される
        assert "このレイヤーは「解析開始」のあとに作られます" in page
        assert page.index("このレイヤーは「解析開始」のあとに作られます") > page.index("function showLayer")

    def test_layers_are_grouped_in_work_order(self):
        page = _page()
        assert "LAYER_GROUPS" in page
        assert "optgroup" in page
        groups = (
            "入力画像（全体）",
            "作業と結果（目元ROI）",
            "診断用（目元ROI）",
            "未解析プレビュー（表示のみ）",
        )
        positions = [page.index(g) for g in groups]
        assert positions == sorted(positions)

    def test_frequently_used_layers_come_before_diagnostics(self):
        page = _page()
        # 実際の作業は Probability→Trimap→Alpha→Product RGBA→再合成 を切り替えて行う。
        # 位置合わせ確認用の roi_b と、手動ROIでは Probability と同一になる difference は後ろ
        work = page.index("['probability', 'trimap', 'alpha', 'product_rgba'")
        assert "'composite_on_bare', 'composite_on_edited'" in page
        assert work < page.index("['roi_a', 'roi_b', 'difference']")

    def test_labels_name_the_image_and_its_role(self):
        page = _page()
        # 「Original」は装着画像を指すのに未装着と誤読されるため使わない
        assert "Original (装着)" not in page
        assert "Aligned (未装着)" not in page
        for label in (
            "目元ROI：装着（抽出元）",
            "目元ROI：未装着（位置合わせ済み）",
            "装着画像（全体）",
            "未装着画像（全体）",
            "未解析プレビュー：装着画像",
        ):
            assert label in page

    def test_reordering_never_drops_a_layer(self):
        page = _page()
        # グループ表に無いレイヤーも落とさない（再開時の roi_b、再合成レイヤー）
        assert "if (!grouped.has(name))" in page

    def test_default_selection_still_follows_the_caller_order(self):
        page = _page()
        # 並びを正規化しても「解析直後は Probability」「Matting 後は最新の成果物」を保つ
        assert "const preferred =" in page
        assert page.index("const preferred =") < page.index("sel.innerHTML = ''")

    def test_placeholder_value_is_never_treated_as_a_layer(self):
        page = _page()
        # 既存コードは [...sel.options].map(o => o.value) でレイヤー一覧を作り直す
        assert ".filter(Boolean)" in page


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
