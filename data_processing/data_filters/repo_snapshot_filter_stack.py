import os.path
import re
import string
from typing import AnyStr

from bs4 import BeautifulSoup

from data_processing.data_filters.repo_snapshot_filter_base import SnapshotFilterBase


class SnapshotFilterStack(SnapshotFilterBase):
    def __init__(self):
        super().__init__()

    def filter_condition(self, filename: str, content: str) -> bool:
        """
        See SnapshotFilterBase.filter_condition for general description

        This implementation in SnapshotFilterStack adds filters from the Stack v2 (https://arxiv.org/pdf/2402.19173.pdf)
        """
        if not self._long_line_condition(filename, content):
            return False

        if not self._autogenerated_condition(filename, content):
            return False

        if not self._alpha_condition(filename, content):
            return False

        if not self._encoded_data_condition(filename, content):
            return False

        if not self._html_condition(filename, content):
            return False

        if not self._text_condition(filename, content):
            return False

        return True

    def _long_line_condition(self, filename: str, content: str) -> bool:
        line_list = content.split('\n')
        char_lens = [len(line) for line in line_list]
        if len(line_list) > 100_000:
            return False
        if sum(char_lens) / len(char_lens) > 150:  # original threshold is 100
            return False
        excluded_extensions = self._get_longline_exceptions()
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in excluded_extensions:
            if max(char_lens) > 1_000:
                return False
        else:
            if max(char_lens) > 100_000:
                return False
        return True

    def _autogenerated_condition(self, filename: str, content: str) -> bool:  # TODO: add go-enry classificator
        line_list = content.split('\n')
        keywords = self._get_autogenerated_keywords()
        for line in line_list[:5]:
            if any(kw in line.lower() for kw in keywords):
                return False
        return True

    def _alpha_condition(self, filename: str, content: str) -> bool:
        # TODO: implement except Motorola 68K Assembly and WebAssembly
        alpha_chars = string.ascii_letters
        total_chars = len(content)
        if total_chars < 1:
            return False
        alpha_chars_count = sum(c in alpha_chars for c in content)
        if alpha_chars_count / total_chars < 0.25:
            return False
        return True

    def _encoded_data_condition(self, filename: str, content: str) -> bool:
        total_matched_length = 0
        for pattern in self._get_encoded_data_patterns():
            matches = pattern.findall(content)  # If it's time-consuming, change to finditer
            if any(len(m) > 1024 for m in matches):
                return False
            total_matched_length += sum(len(m) for m in matches)
        if total_matched_length / len(content) > 0.5:
            return False
        return True

    def _html_condition(self, filename: str, content: str) -> bool:
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() in self._get_html_extensions():
            soup = BeautifulSoup(content, 'html.parser')
            visible_text = soup.get_text()
            visible_text = ' '.join(visible_text.split())
            ratio = len(visible_text) / len(content)
            if ratio >= 0.2 and len(visible_text) > 100:
                return True
            else:
                return False
        return True

    def _text_condition(self, filename: str, content: str) -> bool:
        # TODO: think if it is necessary
        base_filename, file_extension = os.path.splitext(os.path.basename(filename))
        if base_filename.lower() in ['license', 'licence']:
            return False
        if file_extension.lower() in self._get_text_extensions():
            if 'requirements' in base_filename.lower():
                return True
            elif base_filename.lower() in self._get_text_keywords():
                return True
            else:
                return False
        return True

    # TODO: there is also a json/yaml filtering in the Stack paper
    # TODO: filter out hidden files

    @staticmethod
    def _get_longline_exceptions() -> list[str]:
        """
        Naive implementation of the following from the paper:
        'for all languages, excluding HTML, JSON, Markdown, Roff, Roff Manpage, SMT, TeX, Text, and XML'
        """
        return ['.html', '.json', '.md', '.roff', '.man', '.smt', '.tex', '.txt', '.xml']

    @staticmethod
    def _get_autogenerated_keywords() -> list[str]:
        return ['auto-generated', 'autogenerated', 'automatically generated',
                'generated automatically', 'this file is generated']

    @staticmethod
    def _get_encoded_data_patterns() -> list[re.Pattern[AnyStr]]:
        patterns = list()
        patterns.append(re.compile(r'[a-zA-Z0-9+/=\n]{64,}'))
        patterns.append(re.compile(r'(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}'))
        patterns.append(re.compile(r'(?:\\u[0-9a-fA-F]{4}){8,}'))
        return patterns

    @staticmethod
    def _get_html_extensions():
        return ['.html', 'htm']

    @staticmethod
    def _get_text_extensions():
        return ['.txt', '.md', '.rtf', '.rst', '.mdx']

    @staticmethod
    def _get_text_keywords():
        return ['readme', 'notes', 'todo', 'description', 'cmakelists']
