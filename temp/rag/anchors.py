import re
from typing import List, Dict, Tuple

ANCHOR_PATTERNS = [
    r"^\s*[IVXLC]+\.\s.+$",          # I. 회사의 개요
    r"^\s*\d+\.\s.+$",               # 1. 회사의 개요
    r"^\s*\d+\-\d+\.\s.+$",          # 1-1. ...
    r"^\s*제\s*\d+\s*기.*$",         # 제 11 기
    r"^\s*\(\d+\)\s.+$",             # (1) 지배기업의 개요
]

def detect_anchors(text: str) -> List[Tuple[int, str]]:
    lines = text.splitlines()
    anchors = []
    for i, line in enumerate(lines):
        for pat in ANCHOR_PATTERNS:
            if re.match(pat, line.strip()):
                anchors.append((i, line.strip()))
                break
    return anchors

def segment_by_anchors(text: str) -> List[Dict]:
    lines = text.splitlines()
    anchors = detect_anchors(text)
    if not anchors:
        return [{"title": "FULL", "text": text}]
    segments = []
    for idx, (lineno, title) in enumerate(anchors):
        start = lineno
        end = anchors[idx+1][0] if idx+1 < len(anchors) else len(lines)
        seg_text = "\n".join(lines[start:end]).strip()
        segments.append({"title": title, "text": seg_text})
    return segments