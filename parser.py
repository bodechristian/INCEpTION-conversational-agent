import re
import logging

from functioncalls import chat, search_context, check_annotations, summarize_document, classify_span, get_layer, get_scope, annotate, highlight, get_feature, respond

valid_functions = {
    "chat": chat,
    "search_context": search_context,
    "check_annotations": check_annotations,
    "summarize": summarize_document,
    "classify_span": classify_span,
    "get_layer": get_layer,
    "get_scope": get_scope,
    "annotate": annotate,
    "highlight": highlight,
    "get_feature": get_feature,
    "respond": respond
}
VALID_FUNCTIONS_STR = "|".join(valid_functions.keys())


def parse_dollars_lines(input_string: str) -> None:
    """Analyzes and executes functions from a specific syntax.
    e.g
    $1 = get_scope(user_query="annotate every animal as such")
    $2 = classify_span(criteria_query="animal", scope=$1)
    $3 = get_layer(original_user_query="annotate every animal as such")
    $4 = get_feature(original_user_query="annotate every animal as such", layer=$3)
    $5 = annotate(layer=$3, feature=$4, scope=$1, annotation_positions=$2)
    $6 = respond(context="")"""

    cache_return_values = dict()
    regexpr = fr"""\$(\d+)\s*=\s*({
        VALID_FUNCTIONS_STR})\((\w+=(?:".*"|\$\d+)(?:,\s*\w+=(?:"\w+"|\$\d+))*)?\)"""

    lines = input_string.splitlines()
    for line in lines:
        # each line has to match the regex
        if (the_match := re.match(regexpr, line)):
            # extract the name of the function and its parameters from the regex
            # as well as the number of the line (used for cache)
            num, function_to_call, parameters = the_match.groups()
            # analyze the parameters and read them from the cache if nessecary
            json_kwargs = {}
            if parameters:
                parameters = [el.strip() for el in parameters.split(",")]
                for p in parameters:
                    kw, val = p.split("=")
                    if val.startswith("$"):
                        json_kwargs[kw] = cache_return_values[val[1:]]
                    else:
                        json_kwargs[kw] = val.strip('"')
            # call the actual function
            return_value = valid_functions[function_to_call](**json_kwargs)
            # save return value in cache under the respective line number
            if return_value:
                cache_return_values[num] = return_value
            else:  # called function returns None
                cache_return_values[num] = ""  # soll evtl fehler werfen?
    # returns the last cache entry (atm assumed to be response)
    if (el := cache_return_values[str(len(lines))]) != "":
        return el


if __name__ == "__main__":
    TEST_INPUT = """$1 = get_scope(user_query="annotate every animal as such")
$3 = classify_span(criteria_query="animal", scope=$1)
$4 = get_layer(original_user_query="annotate every animal as such")
$5 = get_feature(original_user_query="annotate every animal as such", layer=$4)
$6 = annotate(layer=$4, feature=$5, scope=$1, annotation_positions=$3)
$7 = respond(context="")"""
    logging.basicConfig(level=logging.DEBUG)

    parse_dollars_lines(TEST_INPUT)
