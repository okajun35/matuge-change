class TestRegisterAsset:
    def test_registers_the_extracted_product(self, client, matted_session):
        res = client.post(
            "/api/assets",
            data={"session_id": matted_session, "name": "テストラッシュ", "brand": "matuge"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["name"] == "テストラッシュ"
        assert body["width"] == 48 and body["height"] == 48

    def test_requires_matting_first(self, client, session_id):
        res = client.post("/api/assets", data={"session_id": session_id, "name": "x"})
        assert res.status_code == 409

    def test_unknown_session_returns_404(self, client):
        res = client.post("/api/assets", data={"session_id": "nope", "name": "x"})
        assert res.status_code == 404

    def test_blank_name_returns_400(self, client, matted_session):
        res = client.post("/api/assets", data={"session_id": matted_session, "name": "  "})
        assert res.status_code == 400


class TestBrowseCatalog:
    def test_registered_asset_is_listed_and_fetchable(self, client, matted_session):
        asset_id = client.post(
            "/api/assets", data={"session_id": matted_session, "name": "一覧テスト"}
        ).json()["id"]

        assert any(a["id"] == asset_id for a in client.get("/api/assets").json()["assets"])
        assert client.get(f"/api/assets/{asset_id}").json()["name"] == "一覧テスト"

        image = client.get(f"/api/assets/{asset_id}/image")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"

    def test_unknown_asset_returns_404(self, client):
        assert client.get("/api/assets/00000000-0000-0000-0000-000000000000").status_code == 404


class TestSimilarity:
    def test_similar_returns_other_assets_scored(self, client, matted_session):
        first = client.post(
            "/api/assets", data={"session_id": matted_session, "name": "A"}
        ).json()["id"]
        client.post("/api/assets", data={"session_id": matted_session, "name": "B"})

        res = client.get(f"/api/assets/{first}/similar", params={"limit": 3})
        assert res.status_code == 200
        matches = res.json()["matches"]
        assert first not in [m["id"] for m in matches]
        assert all("similarity" in m for m in matches)
