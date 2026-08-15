import numpy as np
import pytest

from backend.strokes.stroke import BrushTool, Stroke, StrokeSet


class TestStroke:
    def test_from_dict_roundtrip(self):
        raw = {"tool": "fg", "radius": 8, "points": [[1, 2], [3, 4]]}
        stroke = Stroke.from_dict(raw)
        assert stroke.tool is BrushTool.FOREGROUND
        assert stroke.radius == 8
        assert stroke.points == [(1.0, 2.0), (3.0, 4.0)]
        assert stroke.to_dict() == raw

    def test_unknown_tool_is_rejected(self):
        with pytest.raises(ValueError):
            Stroke.from_dict({"tool": "sparkle", "radius": 4, "points": [[0, 0]]})

    def test_empty_points_is_rejected(self):
        with pytest.raises(ValueError):
            Stroke.from_dict({"tool": "bg", "radius": 4, "points": []})

    def test_non_positive_radius_is_rejected(self):
        with pytest.raises(ValueError):
            Stroke.from_dict({"tool": "bg", "radius": 0, "points": [[0, 0]]})


class TestStrokeSet:
    def test_from_payload_parses_all_strokes(self):
        strokes = StrokeSet.from_payload(
            32,
            16,
            [
                {"tool": "fg", "radius": 3, "points": [[4, 4]]},
                {"tool": "bg", "radius": 3, "points": [[20, 8]]},
            ],
        )
        assert len(strokes) == 2
        assert strokes.width == 32 and strokes.height == 16

    def test_rasterize_maps_tools_to_constraint_values(self):
        strokes = StrokeSet.from_payload(
            40,
            40,
            [
                {"tool": "fg", "radius": 3, "points": [[5, 5]]},
                {"tool": "unknown", "radius": 3, "points": [[20, 5]]},
                {"tool": "bg", "radius": 3, "points": [[30, 5]]},
            ],
        )
        constraints = strokes.rasterize()
        assert constraints.shape == (40, 40)
        assert constraints.dtype == np.int8
        assert constraints[5, 5] == 1
        assert constraints[5, 20] == 2
        assert constraints[5, 30] == -1
        assert constraints[35, 35] == 0

    def test_later_strokes_overwrite_earlier_ones(self):
        strokes = StrokeSet.from_payload(
            20,
            20,
            [
                {"tool": "fg", "radius": 5, "points": [[10, 10]]},
                {"tool": "bg", "radius": 5, "points": [[10, 10]]},
            ],
        )
        assert strokes.rasterize()[10, 10] == -1

    def test_polyline_is_filled_between_points(self):
        strokes = StrokeSet.from_payload(40, 40, [{"tool": "fg", "radius": 2, "points": [[5, 20], [35, 20]]}])
        constraints = strokes.rasterize()
        assert constraints[20, 20] == 1

    def test_empty_set_rasterizes_to_zeros(self):
        assert not StrokeSet.from_payload(8, 8, []).rasterize().any()

    def test_payload_roundtrip(self):
        payload = [{"tool": "unknown", "radius": 6, "points": [[1, 1], [2, 2]]}]
        assert StrokeSet.from_payload(10, 10, payload).to_payload() == payload

    def test_invalid_canvas_size_is_rejected(self):
        with pytest.raises(ValueError):
            StrokeSet.from_payload(0, 10, [])
