PAYLOAD = {
    "width": 48,
    "height": 48,
    "strokes": [{"tool": "fg", "radius": 6, "points": [[10, 10], [20, 12]]}],
}


class TestStrokePersistence:
    def test_save_then_restore(self, client, session_id):
        assert client.put(f"/api/sessions/{session_id}/strokes", json=PAYLOAD).status_code == 200
        body = client.get(f"/api/sessions/{session_id}/strokes").json()
        assert body == PAYLOAD

    def test_session_without_strokes_returns_empty_set(self, client, session_id):
        body = client.get(f"/api/sessions/{session_id}/strokes").json()
        assert body["strokes"] == []

    def test_unknown_session_returns_404(self, client):
        assert client.get("/api/sessions/nope/strokes").status_code == 404

    def test_invalid_tool_returns_400(self, client, session_id):
        bad = {**PAYLOAD, "strokes": [{"tool": "glitter", "radius": 3, "points": [[1, 1]]}]}
        assert client.put(f"/api/sessions/{session_id}/strokes", json=bad).status_code == 400

    def test_saved_strokes_drive_the_matte(self, client, session_id):
        strokes = {
            "width": 48,
            "height": 48,
            "strokes": [{"tool": "fg", "radius": 4, "points": [[5, 5]]}],
        }
        client.put(f"/api/sessions/{session_id}/strokes", json=strokes)
        res = client.post("/api/matte", data={"session_id": session_id, "use_saved_strokes": "true"})
        assert res.status_code == 200
        assert "product_rgba" in res.json()["layers"]
