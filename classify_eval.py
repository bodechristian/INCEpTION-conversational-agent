import os
import re
import sys
import utils
import logging
import time
import json
import matplotlib.pyplot as plt

from tqdm import tqdm
from groq import Groq
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent import Agent, CLIENTMODELS
from cerebras.cloud.sdk import Cerebras
from prompts import *


dataset = [("annotate all animals in the document", [('animal', (0, 7)), ('animal', (15, 22)), ('animal', (459, 466)), ('animal', (637, 644)), ('animal', (1091, 1098)), ('animal', (1400, 1407)), ('animal', (1588, 1594)), ('animal', (1596, 1605)), ('animal', (1610, 1628)), ('animal', (1634, 1641)), ('animal', (1925, 1932)), ('animal', (2093, 2100)), ('animal', (2211, 2218)), ('animal',
            (2577, 2584)), ('animal', (2764, 2771)), ('animal', (3099, 3106)), ('animal', (3467, 3474)), ('animal', (3534, 3545)), ('animal', (3675, 3686)), ('animal', (3815, 3822)), ('animal', (3898, 3911)), ('animal', (4014, 4021)), ('animal', (4189, 4196)), ('animal', (4296, 4303)), ('animal', (4511, 4518)), ('animal', (4639, 4646)), ('animal', (4651, 4658)), ('animal', (4829, 4836))])]


def compare(res, gold):
    res_idxs = [idx for _, idx in res]
    gold_idxs = [idx for _, idx in gold]

    cnt = 0
    for el in res_idxs:
        if el in gold_idxs:
            cnt += 1

    tp, fp, tn, fn = 0, 0, 0, 0
    res_idxs_set = set(res_idxs)
    gold_idxs_set = set(gold_idxs)
    for el in res_idxs_set | gold_idxs_set:
        if el in res_idxs_set:
            # in results
            if el in gold_idxs_set:
                # and in gold -> true positive
                tp += 1
            else:
                # but not in gold -> false positive
                fp += 1
        else:
            # not in results
            if el in gold_idxs_set:
                # but was expected -> false negative
                fn += 1
            else:
                # but also wasn't expected -> true negative
                tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall
    return f"Precision: {precision:.2f}\tRecall: {recall:.2f}"


if __name__ == "__main__":
    client = "ollama"
    model = CLIENTMODELS[client]
    log = []

    for client, model, method, chunksize in [
        ('ukp', 'qwen2.5:32b', 'a', 10000),
    ]:
        for prompt, goldannos in dataset:
            # clear loggers
            l = logging.getLogger('functions')
            l.handlers.clear()
            l = logging.getLogger('output')
            l.handlers.clear()

            # create agent
            agent = Agent(mode='planner', client=client, model=model, debug=False, test_classify=True)
            methods = {
                "classify1": agent.functionclass_classify.classify_span1,
                "classify1_noICL": agent.functionclass_classify.classify_span_no_icl,
                "a": agent.functionclass_classify.classify_span_exactwordcheck2,
            }

            # call function
            start_time = time.time()
            result = methods[method](criteria_query=prompt, scope='current document', chunk_size=chunksize)
            duration = time.time() - start_time
            precision, recall = compare(result, goldannos)

            found_words = [(agent.softwareenv.get_current_documenttext()[s:e], cat, s, e) for cat, (s, e) in result]

            log.append({
                "function": method,
                "chunk_size": chunksize,
                "client": client,
                "model": model,
                "nb_annotations_made": len(result),
                "precision": precision,
                "recall": recall,
                "duration": duration,
                "tokens_prompt": agent.nb_tokens_prompt,
                "tokens_completion": agent.nb_tokens_completion,
                "nb_api_calls": agent.nb_api_calls,
                "annotations made": found_words,
            })

    # Convert and write JSON object to file
    with open(os.path.join("test_logs_classify", f"{time.strftime("%Y%m%d-%H%M%S")}.json"), "w") as outfile:
        json.dump(log, outfile)
