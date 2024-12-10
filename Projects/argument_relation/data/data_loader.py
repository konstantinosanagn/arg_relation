# data_loader.py
import json
from typing import List
from argument_relation.data.data_types import Comment

def load_comments(filepath: str) -> List[Comment]:
    """
    Load comments from a jsonlist file, each line containing one comment's data.
    Returns a list of Comment objects.
    """
    comments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                comment = Comment.from_dict(data)
                comments.append(comment)
            except json.JSONDecodeError as e:
                # Handle or log malformed lines
                print(f"Warning: Skipping malformed line. Error: {e}")
                continue
    return comments

