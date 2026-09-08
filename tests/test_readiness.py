from otovision.api import STATE, health, ready


def test_unloaded_model_is_live_but_not_ready():
    original = dict(STATE)
    try:
        STATE.clear()
        STATE["bundle"] = None
        assert health() == {"status": "ok"}
        assert ready().status_code == 503
        STATE["bundle"] = object()
        assert ready().status_code == 200
    finally:
        STATE.clear()
        STATE.update(original)
