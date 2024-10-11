import re
import logging

from os import listdir, getcwd
from os.path import join

if __name__ == "__main__":
    # Removes citation brackets ('[..]') after sentence endings
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
