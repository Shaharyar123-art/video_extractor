"""Player profile (samples/players/<id>/player.json) aur naam matching.

Naam teen jagah aata hai — scoreboard (SURNAME, uppercase), name-card (FULL NAME),
commentary (English / Urdu / Hindi script). Is liye har player ke aliases banate hain
aur fuzzy match karte hain (OCR/Whisper spelling ki ghaltiyan bardasht ho jayein).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp"}


def normalize(text: str) -> str:
    """Lowercase, Latin accents hatao, punctuation hatao.

    Combining marks sirf Latin letters ke baad hataye jate hain — Hindi ki matraen (ि ु ा)
    aur Urdu ke aerab bhi combining hain, unhein hataane se 'क़ादिर' → 'क द र' ban jata tha.
    Hindi nuqta (़) hata dete hain taake 'क़ादिर' aur 'कादिर' barabar hon.
    """
    text = unicodedata.normalize("NFKD", text)
    out: list[str] = []
    for ch in text:
        if ch == "़":  # Devanagari nuqta
            continue
        if unicodedata.combining(ch) and out and out[-1] < "ɐ":
            continue  # Latin accent (é → e)
        out.append(ch)
    text = unicodedata.normalize("NFC", "".join(out)).casefold()
    text = re.sub(r"[^\w\sऀ-ॿ؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Player:
    id: str
    name: str
    jersey_number: str | None = None
    aliases: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)

    @property
    def surname(self) -> str:
        return self.name.split()[-1]

    def all_names(self) -> list[str]:
        """Full name + surname + 'A Surname' + user ke aliases — normalized, unique.
        Bohat chhote (≤2 letters) aliases false matches dete hain, is liye nikal dete hain."""
        parts = self.name.split()
        auto = [self.name, self.surname]
        if len(parts) > 1:
            auto.append(f"{parts[0][0]} {self.surname}")
        seen: dict[str, None] = {}
        for n in [*auto, *self.aliases]:
            nn = normalize(n)
            if len(nn.replace(" ", "")) > 2:
                seen[nn] = None
        return list(seen)

    @classmethod
    def load(cls, folder: str | Path) -> Player:
        folder = Path(folder)
        meta = json.loads((folder / "player.json").read_text(encoding="utf-8"))
        name = meta.get("player_name") or meta.get("display_name") or folder.name.replace("_", " ")
        aliases = list(meta.get("aliases", []))
        for key in ("player_name_urdu", "player_name_hindi"):
            if meta.get(key):
                aliases.append(meta[key])
        jersey = meta.get("jersey_number")
        return cls(
            id=meta.get("player_id") or meta.get("id") or folder.name,
            name=name,
            jersey_number=str(jersey) if jersey not in (None, "") else None,
            aliases=aliases,
            images=sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT),
        )


def window_score(text: str, names: list[str]) -> float:
    """0–100: poora `text` kisi naam jaisa hai? (andar kahin hona kaafi nahi)."""
    t = normalize(text)
    if not t:
        return 0.0
    return max((fuzz.ratio(t, n) for n in names), default=0.0)


def name_score(text: str, names: list[str]) -> float:
    """0–100: `text` (OCR line ya transcript ka hissa) mein koi naam kitna match karta hai.

    - Exact word match → 100
    - Warna fuzzy (partial_ratio) — lekin sirf barabar lambai ke words par, taake 'QADIR'
      ka match 'QADEER' se ho jaye lekin 'AD' jaise tukron se na ho.
    """
    t = normalize(text)
    if not t:
        return 0.0
    words = t.split()
    best = 0.0
    for name in names:
        n_words = name.split()
        k = len(n_words)
        # Exact word-sequence match
        for i in range(len(words) - k + 1):
            if words[i : i + k] == n_words:
                return 100.0
        # Fuzzy: har k-word window ko naam se compare karo
        for i in range(max(1, len(words) - k + 1)):
            window = " ".join(words[i : i + k])
            if abs(len(window) - len(name)) > max(2, len(name) // 3):
                continue
            best = max(best, fuzz.ratio(window, name))
    return best
