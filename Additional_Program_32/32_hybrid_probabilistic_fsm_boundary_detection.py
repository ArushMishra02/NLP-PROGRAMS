# Program 32: Hybrid probabilistic + FSM-based sequence modeling for document boundary detection
import re
text = """UNIVERSITY REPORT
Department of Computer Science
This report describes NLP laboratory work.
1. Introduction
Natural language processing analyzes text.
The system detects boundaries between sections.
CONCLUSION
The method combines heading cues with a finite-state machine."""
lines = [line.strip() for line in text.splitlines() if line.strip()]
def heading_probability(line):
    score = 0.0
    if len(line) < 70: score += .25
    if line.isupper(): score += .35
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", line): score += .35
    if line.endswith(":"): score += .15
    if line.endswith(".") and not line.isupper(): score -= .15
    return max(0.0, min(1.0, score))
state, sections, current = "OUTSIDE_SECTION", [], None
for line in lines:
    probability = heading_probability(line)
    if probability >= .55:
        if current: sections.append(current)
        current = {"heading": line, "body": [], "probability": round(probability, 2)}
        state = "INSIDE_SECTION"
    else:
        if current is None:
            current = {"heading": "Preamble", "body": [], "probability": 0.0}
        current["body"].append(line)
        state = "INSIDE_SECTION"
if current: sections.append(current)
for i, section in enumerate(sections, 1):
    print(f"\n{i}. {section['heading']} (probability={section['probability']})")
    for line in section["body"]: print("  ", line)
