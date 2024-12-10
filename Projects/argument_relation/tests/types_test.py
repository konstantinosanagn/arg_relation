# test_types.py

import unittest
from ..data_types import Comment, Proposition, load_comments_from_jsonlist
import json
from io import StringIO
from unittest.mock import patch

class TestTypes(unittest.TestCase):
    def test_proposition_from_dict(self):
        prop_dict = {
            "id": 1,
            "type": "policy",
            "text": "Test proposition.",
            "reasons": ["0", "2"],
            "evidence": None
        }
        prop = Proposition.from_dict(prop_dict)
        self.assertEqual(prop.id, 1)
        self.assertEqual(prop.type, "policy")
        self.assertEqual(prop.text, "Test proposition.")
        self.assertEqual(prop.reasons, ["0", "2"])
        self.assertEqual(prop.evidence, [])

    def test_comment_from_dict(self):
        comment_dict = {
            "id": 100,
            "propositions": [
                {"id": 0, "type": "fact", "text": "Fact text", "reasons": None, "evidence": None},
                {"id": 1, "type": "policy", "text": "Policy text", "reasons": ["0"], "evidence": None}
            ]
        }
        comment = Comment.from_dict(comment_dict)
        self.assertEqual(comment.id, "100")
        self.assertEqual(len(comment.propositions), 2)
        self.assertEqual(comment.propositions[0].text, "Fact text")
        self.assertEqual(comment.propositions[1].reasons, ["0"])

    @patch("builtins.open", create=True)
    def test_load_comments_from_jsonlist(self, mock_open):
        mock_data = """{"id":1,"propositions":[{"id":0,"type":"fact","text":"Test","reasons":null,"evidence":null}]}\n"""
        mock_open.return_value = StringIO(mock_data)
        comments = load_comments_from_jsonlist("dummy_path.jsonlist")
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].id, "1")
        self.assertEqual(len(comments[0].propositions), 1)
        self.assertEqual(comments[0].propositions[0].text, "Test")

if __name__ == '__main__':
    unittest.main()

