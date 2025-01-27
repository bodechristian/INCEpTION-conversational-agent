import re
import logging

from os import listdir, getcwd
from os.path import join


def clean_wikipedia_articles():
    """Removes citation brackets ('[..]') after sentence endings"""
    # This is done because INCEpTION doesn't recognize the sentence endings otherwise
    for filename in listdir(join(getcwd(), "corpi")):
        if filename.startswith("cleaned"):
            continue
        filepath = join(getcwd(), "corpi", filename)
        with open(filepath, "r", encoding="utf8") as f:
            filedata = f.read()
            new_text = re.sub(r'([.!?:"])(\[\d+\])+', r'\1', filedata)
        with open(join(getcwd(), "corpi", f"cleaned{filename}"), "w", encoding="utf8") as f:
            f.write(new_text)
        logging.info("cleaned %s", filepath)


def get_toolcalls_from_messages(messages):
    toolcalls = [m['name'] for m in messages if isinstance(m, dict) and m['role'] == 'tool']
    if messages[-1]['role'] == 'assistant':
        toolcalls.append('respond')
    return toolcalls


def remove_file_ending(_str):
    return ".".join(_str.split('.')[:-1])


def parse_deepseek_response(_str):
    _str == (re.split(r'think>', _str)[1]).strip()
    if r'\`\`\`' in _str:
        _str == (re.split(r'\`\`\`', _str)[1]).strip()
    return _str


class ParsingException:
    """Used to indicate parsing errors without throwing exceptions"""

    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return "<ParsingException> " + self.message

    def __str__(self):
        return "<ParsingException> " + self.message
