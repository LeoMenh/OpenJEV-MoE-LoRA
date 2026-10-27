"""Fixed, label-free clinical text view for a complementary lexical branch."""
import re

ANCHOR = re.compile(
    r"(?m)^[ \t]*(?:#{1,3}[ \t]*[^\s#]|\d{1,2}[.)][ \t]+[A-Za-z]|"
    r"(?:primary|secondary|discharge|active) diagnoses?\s*:|"
    r"(?:major |operative )?procedures?\s*:)", re.IGNORECASE
)


def clinical_summary(text):
    """Prefix plus short local spans at explicit problems/diagnoses/procedures.

    The rule is fixed across splits and never accepts labels. Overlapping spans
    are merged so adjacent problems do not accidentally duplicate text.
    """
    spans = [(0, min(len(text), 1000))]
    spans.extend((m.start(), min(len(text), m.start() + 320)) for m in ANCHOR.finditer(text))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return "\n".join(text[start:end] for start, end in merged)[:12000]
