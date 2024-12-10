# types.py
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Proposition:
    id: int
    type: str
    text: str
    reasons: List[str]
    evidence: List[str]

    @classmethod
    def from_dict(cls, obj: dict) -> 'Proposition':
        # Handle null for reasons/evidence by providing empty lists
        reasons = obj.get('reasons') or []
        evidence = obj.get('evidence') or []
        return cls(
            id=obj['id'],
            type=obj['type'],
            text=obj['text'],
            reasons=reasons,
            evidence=evidence
        )

@dataclass
class Comment:
    id: str
    propositions: List[Proposition]

    @classmethod
    def from_dict(cls, obj: dict) -> 'Comment':
        propositions_raw = obj.get('propositions', [])
        propositions = [Proposition.from_dict(p) for p in propositions_raw]
        return cls(
            id=str(obj['id']), 
            propositions=propositions
        )

def load_comments_from_jsonlist(filepath: str) -> List[Comment]:
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse the JSON line and convert to a Comment
            data = json.loads(line)
            comment = Comment.from_dict(data)
            comments.append(comment)
    return comments

