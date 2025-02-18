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


def load_dag(dag: dict) -> dict:
    """takes in list representing a DAG and returns dict 
    where keys are functions and their values are functions that have to be completed right before it

    e.g. 
    {
        root: ['get_scope', 'get_layer_and_feature'],
        get_scope: ['classify_span'],
        get_layer_and_feature: ['annotate'],
        classify_span: ['annotate'],
        annotate: ['respond']
    }
    -> {
        root: ['get_scope', 'get_layer_and_feature'],
        get_scope: ['classify_span'],
        get_layer_and_feature: ['annotate'],
        classify_span: ['annotate'],
        annotate: ['respond'],
        respond: ['<END>']
    }"""
    # tasks in values that are not present as a key only have incoming edges
    # thus we add END tokens to them
    a = list(dag.values())
    tasks = [item for sublist in a for item in sublist]
    for task in set(tasks):
        if not task in dag.keys():
            # add end token
            dag[task] = ["<END>"]
    return dag


def eval_functions_dag(functions: list, dag: dict[str, list]):
    """Evaluates an ordered list of functions according to a dag
    returns validity and number of unnecessary functions
    dag looks like:
    {
        root: [getscope, getlayer],
        getscope: [classify],
        getlayer: [annotate],
        classify: [annotate],
        annotate: [respond],
        respond: [<END>]
    }"""
    current_nodes = dag["root"]
    nb_unused_funcs = 0
    for func in functions:
        if not func in current_nodes:
            # function was "unnecessary", didN't move pointer further along in dag
            nb_unused_funcs += 1
        else:
            # function moved pointer further ahead
            current_nodes = [el for el in current_nodes if el != func]  # remove all funcs
            current_nodes.extend(dag[func])
    valid = (current_nodes == ["<END>"])
    return valid, nb_unused_funcs


def flatten_dag(dag) -> str:
    """dag:
        {
        root: [getscope, getlayer],
        getscope: [classify],
        getlayer: [annotate],
        classify: [annotate],
        annotate: [respond],
        respond: [<END>]
    }
    ->
    'getscope -> classify, getlayer -> annotate -> respond'
    """
    # get last nodes (incoming edge but no outgoing)
    last_nodes = []
    for k, v in dag.items():
        if v == ["<END>"]:
            last_nodes.append(k)
    # go backwards
    lst = [last_nodes]
    while True:
        lst_temp = []
        for k, vs in dag.items():
            # go backwards an edge
            for v in vs:
                if v in lst[0] and k != "root":
                    lst_temp.append(k)
        if lst_temp != []:
            # add all the previous nodes
            lst.insert(0, lst_temp)
        else:
            # temporary list is empty, so nothing was added and we re done
            break

    # [[getscope], [classify, getlayer], [annotate], [respond]]
    # getscope -> classify, getlayer -> annotate -> respond
    _str = ' -> '.join([', '.join(sublist) for sublist in lst])
    return _str


class ParsingException:
    """Used to indicate parsing errors without throwing exceptions"""

    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return "<ParsingException> " + self.message

    def __str__(self):
        return "<ParsingException> " + self.message


if __name__ == "__main__":
    d = {
        "root": ['get_scope', 'get_layer_and_feature'],
        "get_scope": ['classify_span'],
        "get_layer_and_feature": ['annotate'],
        "classify_span": ['annotate'],
        "annotate": ['respond']
    }
    d = load_dag(d)
    print(d)
    print(eval_functions_dag(dag=d, functions=["get_scope", "search_context",
          "get_layer_and_feature", "classify_span", "annotate", "respond"]))
    print(flatten_dag(d))
