"""
Module containing parsing classes for a Llama-generated response.
"""

from abc import ABC, abstractmethod
from typing import List
import re

class BaseResponseParser(ABC):
    """
    Abstract base class for parsing an assistant response.
    """
    def __init__(self, answer_token: str, answer_format: str):
        """
        Attributes:
            answer_token (str): The token placeholder for an answer.
            answer_format (str): A format string for the expected assistant response.
        """
        self.answer_token = answer_token
        self.answer_format = answer_format

    @abstractmethod
    def get_parsed_response(self, response: str) -> str:
        """
        Parses an assistant response for a particular subsequence.
        Args:
            response (str): The assistant response to parse.
        Returns:
            A processed subsequence of the response.
        """
        raise NotImplementedError

class PropositionResponseParser(BaseResponseParser):
    """
    Parses the proposition classification from an assistant response.
    """
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='Classification: {}'):
        """
        See parent.
        """
        super().__init__(answer_token, answer_format)

    def _get_answer_using_regex_match(self, pattern: str, response: str) -> str:
        match: re.Match = re.search(pattern, response, re.IGNORECASE)
        result = match.group(1) if match else ""
        return result

    def get_parsed_response(self, response: str) -> str:
        """
        See parent.
        """
        # BUG: This info is repeated in multiple places
        proposition_types = set(['fact', 'testimony', 'policy', 'value', 'reference'])
        patterns = [
                # Search for answer in quotes
                r'.*?"(\w+)"',
                # Search for answer in answer_format
                '.*?{}'.format(self.answer_format.replace('{}', r'(\w+)')),
                '.*?{}'.format(self.answer_format.replace('{}', r'"(\w+)"'))
                ]
        print(f"Parsing assistant response: {response}")
        for pattern in patterns:
            search_res = self._get_answer_using_regex_match(pattern, response)
            if search_res:
                break
        else:
            return ""
        # Llama's answer will be somewhere after the search_token
        possible_ans = search_res.strip().lower()
        print(f"Possible parsed response: {possible_ans}")
        # BUG: Should throw?
        return possible_ans if possible_ans in proposition_types else ""

class SupportResponseParser(BaseResponseParser):
    def __init__(self,
                 answer_token: str='<answer>',
                 answer_format: str='Answer: {}'):
        super().__init__(answer_token, answer_format)
        
    def _get_answer_using_regex_match(self, pattern: str, response: str) -> str:
        match: re.Match = re.search(pattern, response, re.IGNORECASE)
        result = match.group(1) if match else ""
        return result

    def get_parsed_response(self, response: str) -> str:
        """
        See parent.
        """
        # Define the possible answers
        proposition_types = set(['yes', 'no'])

        # Define regex patterns to capture the response
        patterns = [
            r'<answer>\s*(yes|no)\s*</answer>',  # Capture 'yes' or 'no' within <answer> tags
            r'<answer>\s*(Yes|No)\s*</answer>',  # Capture 'Yes' or 'No' within <answer> tags (case insensitive)
            r'\b(yes|no)\b',  # Capture 'yes' or 'no' as standalone words
            r'\b(Yes|No)\b',  # Capture 'Yes' or 'No' as standalone words (case insensitive)
            r'".*?\b(yes|no)\b.*?"',  # Capture 'yes' or 'no' within quotes
            r'".*?\b(Yes|No)\b.*?"',  # Capture 'Yes' or 'No' within quotes (case insensitive)
            r'\b(support|attack)\b',  # Capture 'support' or 'attack' as standalone words
            r'".*?\b(support|attack)\b.*?"'  # Capture 'support' or 'attack' within quotes
        ]

        print(f"Parsing assistant response: {response}")

        # Try matching the response with each pattern
        for pattern in patterns:
            search_res = self._get_answer_using_regex_match(pattern, response)
            if search_res:
                print(f"Matched with pattern: {pattern}")
                break
        else:
            print("No match found.")
            return "unknown"

        # Normalize the response
        possible_ans = search_res.strip().lower()
        print(f"Possible parsed response: {possible_ans}")

        return possible_ans if possible_ans in proposition_types else "unknown"

# Usage example
if __name__ == '__main__':
    parser = SupportResponseParser()
    response = 'The answer is <answer>Yes</answer>.'
    print(parser.get_parsed_response(response))  # Expected output: 'yes'