import re
import sys
import logging

from functioncalls import Agentfunctions
from mock_annotation_tool import MockAnnotationTool


class Dollarparser():

    def __init__(self, functionclass: Agentfunctions) -> None:
        self.functionclass = functionclass
        self.valid_functions_str = "|".join(
            functionclass.valid_functions.keys())

    def analyze_functions(self, input_string: str):
        regexpr = fr"""\$(\d+)\s*=\s*({
            self.valid_functions_str})\((\w+=(?:".*"|\$\d+)(?:,\s*\w+=(?:"\w+"|\$\d+))*)?\)"""

        lines = input_string.splitlines()
        functions = []
        for line in lines:
            # each line has to match the regex
            if (the_match := re.match(regexpr, line)):
                # extract the name of the function and its parameters from the regex
                # as well as the number of the line (used for cache)
                num, function_to_call, parameters = the_match.groups()
                functions.append((num, function_to_call, parameters))
        return functions

    def call_functions(self, functions):
        try:
            cache_return_values = dict()
            for num, function_to_call, parameters in functions:
                # analyze the parameters and read them from the cache if nessecary
                json_kwargs = {}
                if parameters:
                    # regex splits at commas that are not in quotes
                    # old: r'(?!\B"[^"]*),\s?(?![^"]*"\B)' <- failed at empty quotes as a parameter, new one was created with chatgpt help
                    parameters = re.split(
                        r',\s?(?=(?:(?:[^"]*"){2})*[^"]*$)', parameters)
                    for p in parameters:
                        kw, val = p.split("=")
                        if val.startswith("$"):
                            json_kwargs[kw] = cache_return_values[val[1:]]
                        else:
                            json_kwargs[kw] = val.strip('"')
                # call the actual function
                return_value = self.functionclass.valid_functions[function_to_call](**json_kwargs)
                # save return value in cache under the respective line number
                if return_value:
                    cache_return_values[num] = return_value
                else:  # called function returns None
                    cache_return_values[num] = ""  # soll evtl fehler werfen?
            # returns the last cache entry (assumed to be response)
            if (el := cache_return_values[str(len(functions))]) != "":
                return el
        except Exception as error:
            # Maybe TODO: create custom exception Class
            logger = logging.getLogger("output")
            logger.debug(error)
            return "Unable to parse LLM response"

    def analyze_and_execute_dollar_syntax(self, input_string: str) -> None:
        """Analyzes and executes functions from a specific syntax.
        e.g
        $1 = get_scope(user_query="annotate every animal as such")
        $2 = classify_span(criteria_query="animal", scope=$1)
        $3 = get_layer(original_user_query="annotate every animal as such")
        $4 = get_feature(original_user_query="annotate every animal as such", layer=$3)
        $5 = annotate(layer=$3, feature=$4, scope=$1, annotation_positions=$2)
        $6 = respond(context="")"""

        functions = self.analyze_functions(input_string)
        last_output = self.call_functions(functions)
        return last_output


if __name__ == "__main__":
    TEST_INPUT = """$1 = get_scope(user_query="annotate every animal as such")
$3 = classify_span(criteria_query="animal", scope=$1)
$4 = get_layer(original_user_query="annotate every animal as such")
$5 = get_feature(original_user_query="annotate every animal as such", layer=$4)
$6 = annotate(layer=$4, feature=$5, scope=$1, annotation_positions=$3)
$7 = respond(context="")"""
    logger = logging.getLogger("functions")
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout)

    softwareenv = MockAnnotationTool()
    f = Agentfunctions(softwareenv)
    parser = Dollarparser(f)
    parser.analyze_and_execute_dollar_syntax(TEST_INPUT)
