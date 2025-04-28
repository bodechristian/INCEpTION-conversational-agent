import re
import logging
import json

from collections import defaultdict
from function_schema import get_function_schema
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
    _lst = re.split(r'think>', _str)
    if len(_lst) > 2:
        _str = _lst[2].strip()
    if r'\`\`\`' in _str:
        _str = (re.split(r'\`\`\`', _str)[1]).strip()
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
        respond: []
    }"""
    # tasks in values that are not present as a key only have incoming edges
    # still create an edge for them to an empty list
    a = list(dag.values())
    tasks = [item for sublist in a for item in sublist]
    for task in set(tasks):
        if not task in dag.keys():
            dag[task] = []
    return dag


def create_parent_dict(dag: dict[str, list[str]]):
    """returns a dictionary where the keys are the nodes and the values are the list of their children
    input:
    {
        root: [getscope, getlayer],
        getscope: [classify],
        getlayer: [annotate],
        classify: [annotate],
        annotate: [respond],
        respond: []
    }
    output:
    {
        'get_scope': [],
        'get_layer': [], 
        'classify': ['get_scope'], 
        'annotate': ['get_layer', 'classify'], 
        'respond': ['annotate']
    }"""
    result_dict = defaultdict(list)
    for k, v in dag.items():
        if k == "root":
            for node in v:
                result_dict[node] = []
        else:
            for node in v:
                result_dict[node].append(k)
    return (result_dict)


def eval_functions_dag(functions: list[str], dag: dict[str, list[str]]):
    """Evaluates an ordered list of functions according to a dag
    returns validity and number of unnecessary functions
    DAG looks like:
    {
        root: [getscope, getlayer],
        getscope: [classify],
        getlayer: [annotate],
        classify: [annotate],
        annotate: [respond],
        respond: []
    }"""
    current_nodes = dag["root"]
    dict_children = create_parent_dict(dag)
    nb_unused_funcs = 0
    for func in functions:
        if not func in current_nodes or dict_children[func] != []:
            # function was "unnecessary", did not move pointer further along in dag
            nb_unused_funcs += 1
        else:
            # function moved pointer further ahead
            current_nodes = [el for el in current_nodes if el != func]  # remove all funcs
            current_nodes.extend(dag[func])
            for new_leaf_node in dag[func]:
                dict_children[new_leaf_node].remove(func)
    is_valid = (current_nodes == [])
    return is_valid, nb_unused_funcs


def flatten_dag(dag) -> str:
    """dag:
        {
        root: [getscope, getlayer],
        getscope: [classify],
        getlayer: [annotate],
        classify: [annotate],
        annotate: [respond],
        respond: []
    }
    ->
    'getscope -> classify, getlayer -> annotate -> respond'
    """
    # get last nodes (incoming edge but no outgoing)
    last_nodes = []
    for k, v in dag.items():
        if v == []:
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


def detect_functions_parampred(funcs_compound):
    compound_map = {
        "annotate": ["get_layer_and_feature", "annotate"],
        "classify_span": ["get_scope", "classify_span"],
        "highlight": ["highlight"],
        "summarize_document": ["get_scope", "summarize_document"],
        "search_annotations": ["search_annotations"],
        "search_context": ["search_context"],
        "respond": ["respond"],
    }

    return [func for funcs_atomic in funcs_compound for func in compound_map[funcs_atomic]]


def detect_functions_compound(funcs_compound):
    compound_map = {
        "annotate": ["get_scope", "get_layer_and_feature", "classify_span", "annotate"],
        "highlight": ["get_scope", "classify_span", "highlight"],
        "summarize_document": ["get_scope", "summarize_document"],
        "search_annotations": ["search_annotations"],
        "search_context": ["search_context"],
        "respond": ["respond"],
    }

    return [func for funcs_atomic in funcs_compound for func in compound_map[funcs_atomic]]


def create_tools_schema(lst_of_functions: list[callable]):
    """Turns list of functions into a json schema that can be passed to tools= api field for function calling
    E.g., get list of functions from prototype like this:
        ag = Agent(mode="sequential", toolcalling_functions=True)
        lst_of_functions = ag.functionclass.valid_functions.values()"""
    schema = [
        {
            "type": "function",
            "function": get_function_schema(f)
        }
        for f in lst_of_functions]
    # print(json.dumps(schema, indent=2))
    return schema


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
    # print(d)
    print(eval_functions_dag(dag=d, functions=["get_scope",
          "classify_span",  "annotate",  "get_layer_and_feature", "annotate", "respond"]))
    # print(flatten_dag(d))

    # lst = ["classify_span", "annotate", "respond"]
    # a = detect_functions_parampred(lst)
    # print(a)
    # c = ['get_scope',  'classify_span', 'get_layer_and_feature', 'annotate', 'respond']
    # b = {
    #     "root": ['get_scope', 'get_layer_and_feature'],
    #     "get_scope": ['classify_span'],
    #     "get_layer_and_feature": ['annotate'],
    #     "classify_span": ['annotate'],
    #     "annotate": ['respond'],
    #     "respond": ['<END>']
    # }
    # print(eval_functions_dag(c, b))
