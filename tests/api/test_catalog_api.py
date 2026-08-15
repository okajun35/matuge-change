import uuid


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


class TestCatalogPaging:
    def _register(self, client, session, name):
        return client.post("/api/assets", data={"session_id": session, "name": name}).json()["id"]

    def test_page_reports_total_limit_and_offset(self, client, matted_session):
        for name in ("ページ1", "ページ2", "ページ3"):
            self._register(client, matted_session, name)
        body = client.get("/api/assets?limit=2&offset=0").json()
        assert len(body["assets"]) == 2
        assert body["total"] >= 3
        assert body["limit"] == 2 and body["offset"] == 0

    def test_second_page_differs_from_the_first(self, client, matted_session):
        for name in ("めくり1", "めくり2", "めくり3"):
            self._register(client, matted_session, name)
        first = {a["id"] for a in client.get("/api/assets?limit=2&offset=0").json()["assets"]}
        second = {a["id"] for a in client.get("/api/assets?limit=2&offset=2").json()["assets"]}
        assert not (first & second)

    def test_query_narrows_the_result(self, client, matted_session):
        token = uuid.uuid4().hex[:8]
        self._register(client, matted_session, f"検索ヒット-{token}")
        self._register(client, matted_session, "対象外 フラット")
        body = client.get(f"/api/assets?q={token}").json()
        assert [a["name"] for a in body["assets"]] == [f"検索ヒット-{token}"]
        assert body["total"] == 1


class TestSimilarity:
    def test_similar_returns_other_assets_scored(self, client, matted_session):
        first = client.post("/api/assets", data={"session_id": matted_session, "name": "A"}).json()["id"]
        client.post("/api/assets", data={"session_id": matted_session, "name": "B"})

        res = client.get(f"/api/assets/{first}/similar", params={"limit": 3})
        assert res.status_code == 200
        matches = res.json()["matches"]
        assert first not in [m["id"] for m in matches]
        assert all("similarity" in m for m in matches)
