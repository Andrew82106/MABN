import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from model7 import parse_items, token_offsets, MODEL
from transformers import AutoTokenizer


class ParserTests(unittest.TestCase):
    def test_three_items_and_punctuation_end(self):
        text = '1. Alpha was founded in 1990.\n2. Beta was founded in 2000.\n3. Alpha was earlier.'
        items, report = parse_items(text, 'q')
        self.assertTrue(report['all_items_parse_ok'])
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(text[item['start']:item['end']], item['text'])
            self.assertTrue(text[item['last_content_character']].isalnum())

    def test_external_single_item_plain_text(self):
        text = '  The policy was announced in 2025.\n'
        items, _ = parse_items(text, 'external', expected_items=1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['text'], text.strip())
        self.assertTrue(items[0]['parse_ok'])

    def test_missing_duplicate_and_truncated_are_visible(self):
        items, _ = parse_items('1. First.\n1. Duplicate.\n3. Last.', 'q')
        self.assertFalse(items[0]['parse_ok'])
        self.assertEqual(items[1]['parse_reason'], 'missing_number')
        items, _ = parse_items('1. First.\n2. Second.\n3. Trunc', 'q', truncated=True)
        self.assertFalse(items[2]['parse_ok'])

    def test_exact_utf8_offsets(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        text = '1. Montréal — 2025.\n2. 北京。'
        ids = tokenizer.encode(text, add_special_tokens=False)
        offsets = token_offsets(tokenizer, ids, text)
        for char_index, char in enumerate(text):
            self.assertTrue(any(a <= char_index < b for a, b in offsets), (char_index, char))
        self.assertEqual(int(offsets[-1, 1]), len(text))


if __name__ == '__main__':
    unittest.main()
