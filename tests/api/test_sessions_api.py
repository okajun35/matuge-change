class TestGetSession:
    def test_returns_available_layers(self, client, session_id):
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["width"] == 48 and body["height"] == 48
        assert body["layers"] == ["roi_a"]

    def test_layers_grow_after_matting(self, client, matted_session):
        layers = client.get(f"/api/sessions/{matted_session}").json()["layers"]
        assert layers == ["roi_a", "trimap", "alpha", "product_rgba"]

    def test_unknown_session_returns_404(self, client):
        assert client.get("/api/sessions/nope").status_code == 404
