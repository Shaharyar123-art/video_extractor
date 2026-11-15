import json

from pipeline.players import Player, name_score, normalize


def make_player(tmp_path, **meta):
    d = tmp_path / "abdul_qadir"
    d.mkdir()
    (d / "player.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (d / "a.jfif").write_bytes(b"")
    (d / "notes.txt").write_text("x")
    return Player.load(d)


def test_load_user_format(tmp_path):
    p = make_player(
        tmp_path,
        player_id="abdul_qadir",
        player_name="Abdul Qadir",
        jersey_number="17",
        player_name_urdu="عبدالقادر",
        aliases=["A. Qadir"],
    )
    assert p.id == "abdul_qadir"
    assert p.jersey_number == "17"
    assert [i.name for i in p.images] == ["a.jfif"]
    names = p.all_names()
    assert {"abdul qadir", "qadir", "a qadir", "عبدالقادر"} <= set(names)


def test_normalize():
    assert normalize("  QADIR!! ") == "qadir"
    assert normalize("Abdul-Qadir") == "abdul qadir"


NAMES = ["abdul qadir", "qadir", "a qadir"]


def test_exact_scoreboard_match():
    assert name_score("QADIR", NAMES) == 100
    assert name_score("17 ABDUL QADIR", NAMES) == 100


def test_fuzzy_ocr_error():
    assert name_score("QADlR", NAMES) >= 80  # OCR ne i ko l parha


def test_no_false_match():
    assert name_score("USMAN", NAMES) < 60
    assert name_score("RAZA", NAMES) < 60
    assert name_score("AD", NAMES) < 60


def test_commentary_sentence():
    assert name_score("and Qadir drives that beautifully through cover", NAMES) == 100


def test_normalize_keeps_hindi_and_urdu():
    assert normalize("अब्दुल क़ादिर") == normalize("अब्दुल कादिर")
    assert " " in normalize("अब्दुल क़ादिर") and len(normalize("अब्दुल क़ादिर").split()) == 2
    assert normalize("عبدالقادر") == "عبدالقادر"
    assert normalize("Café") == "cafe"
