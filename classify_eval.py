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


dataset = [("every animal", "wikipedia_cheetah", [('animal', (0, 7)), ('animal', (15, 22)), ('animal', (24, 40)), ('animal', (53, 56)), ('animal', (78, 84)), ('animal', (459, 466)), ('animal', (637, 644)), ('animal', (1091, 1098)), ('animal', (1152, 1156)), ('animal', (1400, 1407)), ('animal', (1588, 1594)), ('animal', (1596, 1605)), ('animal', (1610, 1628)), ('animal', (1634, 1641)), ('animal', (1919, 1923)), ('animal', (1925, 1932)), ('animal', (1933, 1937)), ('animal', (2093, 2100)), ('animal', (2211, 2218)), ('animal',
            (2577, 2584)), ('animal', (2764, 2771)), ('animal', (2790, 2805)), ('animal', (2789, 2806)), ('animal', (2798, 2805)), ('animal', (2875, 2883)), ('animal', (3099, 3106)), ('animal', (3467, 3474)), ('animal', (3479, 3485)), ('animal', (3518, 3532)), ('animal', (3525, 3532)), ('animal', (3534, 3545)), ('animal', (3652, 3659)), ('animal', (3645, 3659)), ('animal', (3644, 3660)), ('animal', (3675, 3686)), ('animal', (3815, 3822)), ('animal', (3898, 3911)), ('animal', (3954, 3962)),
    ('animal', (4014, 4021)), ('animal', (4049, 4062)), ('animal', (4120, 4129)), ('animal', (4189, 4196)), ('animal', (4215, 4222)), ('animal', (4296, 4303)), ('animal', (4440, 4451)), ('animal', (4503, 4519)), ('animal', (4504, 4518)), ('animal', (4511, 4518)), ('animal', (4639, 4646)), ('animal', (4651, 4658)), ('animal', (4660, 4675)), ('animal', (4728, 4746)), ('animal', (4729, 4745)), ('animal', (4737, 4745)), ('animal', (4787, 4794)), ('animal', (4829, 4836)), ('animal', (5116, 5119)), ('animal', (5158, 5161))]),
    ("politician", "cleanedwikipedia_2016_pres_election", [('Donald Trump', (207, 219)), ('Trump', (214, 219)), ('Indiana governor Mike Pence', (224, 251)), ('Mike Pence', (241, 251)), ('First Lady of the United States Hillary Clinton', (317, 364)), ('Hillary Clinton', (349, 364)), ('Clinton', (357, 364)), ('Tim Kaine', (403, 412)), ('Incumbent Democratic president Barack Obama', (817, 860)), ('Democratic president Barack Obama', (827, 860)), ('president Barack Obama', (838, 860)), ('Barack Obama', (848, 860)), ('Clinton', (991, 998)), ('U.S. senator Bernie Sanders', (1027, 1054)), ('senator Bernie Sanders', (1032, 1054)), ('Bernie Sanders', (1040, 1054)), ('Sanders', (1047, 1054)), ('Trump', (1167, 1172)), ('Cruz', (1298, 1302)), ('Marco Rubio', (1307, 1318)), ('Rubio', (1313, 1318)), ('John Kasich', (1330, 1341)), ('Kasich', (1335, 1341)), ('Jeb Bush', (1346, 1354)), ('Bush', (1350, 1354)), ("Trump's", (1380, 1387)), ('Trump', (1380, 1385)), ("Trump's", (1626, 1633)), ('Trump', (1626, 1631)), ('Clinton', (1657, 1664)), ('Trump', (1722, 1727)), ("president Barack Obama's", (1839, 1863)), ('president Barack Obama', (1839, 1861)), ("Barack Obama's", (1849, 1863)), ('Barack Obama', (1849, 1861)), ('Trump', (2042, 2047)), ('Clinton', (2255, 2262)), ("Clinton's", (2255, 2264)), ('Clinton', (2580, 2587)), ('Clinton', (2676, 2683)), ('Trump', (2741, 2746)), ('Trump', (2860, 2865)), ('Trump', (3004, 3009)), ("Trump's", (3130, 3137)), ('Trump', (3130, 3135)), ('Clinton', (3197, 3204)), ("Clinton's", (3197, 3206)), ('Sanders', (3263, 3270)), ('Trump', (3271, 3276)), ('Bernie Sanders', (3314, 3328)), ('Sanders', (3321, 3328)), ('Trump', (3354, 3359)), ('Clinton', (3393, 3400)), ('Trump', (3446, 3451)), ('Clinton', (3466, 3473)), ('Trump', (3475, 3480)), ('Trump', (3564, 3569)), ('Libertarian nominee Gary Johnson', (3788, 3820)), ('Gary Johnson', (3808, 3820)), ('Ross Perot', (3932, 3942)), ('Green Party nominee Jill Stein', (3962, 3992)), ('Jill Stein', (3982, 3992)), ('Independent candidate Evan McMullin', (4037, 4072)), ('Evan McMullin', (4059, 4072)), ('Secretary Clinton', (4439, 4456)), ('Clinton', (4449, 4456)), ('Trump', (4587, 4592)), ("Trump's", (4716, 4723)), ('Trump', (4716, 4721)), ('Trump', (4827, 4832)), ('Barack Obama', (4991, 5003)), ('President Barack Obama', (6015, 6037)), ('Barack Obama', (6025, 6037)), ('Clinton', (7209, 7216)), ('former Florida Governor Jeb Bush', (7221, 7253)), ('Jeb Bush', (7245, 7253)), ('Bush', (7249, 7253)), ('New Jersey Governor Chris Christie', (7300, 7334)), ('Chris Christie', (7320, 7334)), ('Christie', (7326, 7334)), ('Senator Cory Booker', (7339, 7358)), ('Cory Booker', (7347, 7358)), ('Cruz', (7574, 7578)), ('Perry', (7816, 7821)), ('Walker', (7823, 7829)), ('Jindal', (7831, 7837)), ('Graham', (7839, 7845)), ('Pataki', (7851, 7857)), ('Trump', (7931, 7936)), ('Cruz', (7955, 7959)), ('Huckabee', (7973, 7981)), ('Paul', (7983, 7987)), ('Santorum', (7993, 8001)), ('Trump', (8087, 8092)), ('Christie', (8123, 8131)), ('Fiorina', (8133, 8140)), ('Gilmore', (8146, 8153)), ('Bush', (8174, 8178)), ('Trump', (8223, 8228)), ('Rubio', (8230, 8235)), ('Cruz', (8241, 8245)), ('Rubio', (8328, 8333)), ('Cruz', (8370, 8374)), ('Trump', (8430, 8435)), ('Carson', (8501, 8507)), ('Kasich', (8596, 8602)), ('Trump', (8655, 8660)), ('Rubio', (8699, 8704)), ('Trump', (8838, 8843)), ('Cruz', (8845, 8849)), ('Kasich', (8855, 8861)), ('Cruz', (8863, 8867)), ('Trump', (8969, 8974)), ('Trump', (9032, 9037)), ('Cruz', (9312, 9316)), ('Kasich', (9325, 9331)), ('Trump', (9363, 9368)), ('Trump', (9589, 9594)), ('Trump', (9679, 9684)), ("Trump's", (9811, 9818)), ('Trump', (9811, 9816)), ('Trump', (9921, 9926)), ('Trump', (10098, 10103)), ('Trump', (10411, 10416)), ('Trump', (10465, 10470)), ('Cruz', (10472, 10476)), ('Rubio', (10478, 10483)), ('Kasich', (10488, 10494)), ('Trump', (10531, 10536)), ('Cruz', (10583, 10587)), ('Trump', (10728, 10733)), ('Trump', (10919, 10924)), ('New Jersey Governor Chris Christie', (10950, 10984)), ('Chris Christie', (10970, 10984)), ('Christie', (10976, 10984)), ('Newt Gingrich', (11014, 11027)), ('Gingrich', (11019, 11027)), ('Senator Jeff Sessions', (11042, 11063)), ('Jeff Sessions', (11050, 11063)), ('Mary Fallin', (11098, 11109)), ('Bob Corker', (11176, 11186)), ('Tom Cotton', (11237, 11247)), ('Joni Ernst', (11263, 11273)), ('Indiana governor Mike Pence', (11289, 11316)), ('Mike Pence', (11306, 11316)), ('Trump', (11371, 11376)), ('Trump', (11536, 11541)), ('Christie', (11605, 11613)), ('Gingrich', (11615, 11623)), ('Trump', (11699, 11704)), ('Trump', (11745, 11750)), ('Hillary Clinton', (12141, 12156)), ('Clinton', (12149, 12156)), ('Clinton', (12448, 12455)), ('Bernie Sanders', (12579, 12593)), ('Sanders', (12586, 12593)), ('Clinton', (12810, 12817)), ('Sanders', (12822, 12829)), ("former governor of Maryland Martin O'Malley", (12848, 12891)), ("Martin O'Malley", (12876, 12891)), ("O'Malley", (12883, 12891)), ('former independent governor and Republican senator of Rhode Island Lincoln Chafee', (12976, 13057)), ('Republican senator of Rhode Island Lincoln Chafee', (13008, 13057)), ('Lincoln Chafee', (13043, 13057)), ('former Virginia senator Jim Webb', (13083, 13115)), ('Jim Webb', (13107, 13115)), ('Webb', (13111, 13115)), ('former Harvard law professor Lawrence Lessig', (13142, 13186)), ('Lawrence Lessig', (13171, 13186)), ('Lessig', (13180, 13186)), ('Webb', (13233, 13237)), ('Vice President Joe Biden', (13339, 13363)), ('Joe Biden', (13354, 13363)), ('Lessig', (13823, 13829)), ('Clinton', (13871, 13878)), ("O'Malley", (13880, 13888)), ('Sanders', (13894, 13901)), ('Clinton', (13958, 13965)), ('Sanders', (14019, 14026)), ("O'Malley", (14064, 14072)), ('Sanders', (14134, 14141)), ('Clinton', (14250, 14257)), ('Clinton', (14472, 14479)), ('Sanders', (14591, 14598)), ('Sanders', (14706, 14713)), ('Clinton', (14795, 14802)), ('Sanders', (14916, 14923)), ('Clinton', (15009, 15016)), ('Clinton', (15094, 15101)), ('Sanders', (15194, 15201)), ('Clinton', (15316, 15323)), ('Clinton', (15362, 15369)), ('Sanders', (15564, 15571)), ('Sanders', (15617, 15624)), ('Clinton', (15735, 15742)), ('Clinton', (15865, 15872)), ('Clinton', (16010, 16017)), ('Clinton', (16288, 16295)), ('Sanders', (16423, 16430)), ('Clinton', (16466, 16473)), ('Clinton', (16583, 16590)), ('Sanders', (16695, 16702)), ('Clinton', (16827, 16834)), ('Sanders', (16879, 16886)), ('Sanders', (16917, 16924)), ('Clinton', (17060, 17067)), ('Trump', (17078, 17083)), ('Clinton', (17140, 17147)), ('Sanders', (17162, 17169)), ('Sanders', (17274, 17281)), ('Clinton', (17300, 17307)), ('Sanders', (17367, 17374)), ('Clinton', (17433, 17440)), ('Lessig', (17708, 17714)), ('Clinton', (17854, 17861)), ('Clinton', (18033, 18040)), ('Sanders', (18151, 18158)), ('Clinton', (18182, 18189)), ('Clinton', (18270, 18277)), ("Clinton's", (18270, 18279)), ('Senator Cory Booker', (18346, 18365)), ('Cory Booker', (18354, 18365)), ('Senator Sherrod Brown', (18383, 18404)), ('Sherrod Brown', (18391, 18404)), ('Mayor of Los Angeles Eric Garcetti', (18482, 18516)), ('Eric Garcetti', (18503, 18516)), ('Senator Tim Kaine', (18534, 18551)), ('Tim Kaine', (18542, 18551)), ('Labor Secretary Tom Perez', (18567, 18592)), ('Tom Perez', (18583, 18592)), ('Representative Tim Ryan', (18608, 18631)), ('Tim Ryan', (18623, 18631)), ('Senator Elizabeth Warren', (18647, 18671)), ('Elizabeth Warren', (18655, 18671)), ('Clinton', (18723, 18730)), ('Secretary of Agriculture Tom Vilsack', (18752, 18788)), ('Tom Vilsack', (18777, 18788)), ('retired Admiral James Stavridis', (18790, 18821)), ('James Stavridis', (18806, 18821)), ('Governor John Hickenlooper', (18827, 18853)), ('John Hickenlooper', (18836, 18853)), ('Clinton', (18921, 18928)), ('Clinton', (19072, 19079)), ('Senator Tim Kaine', (19110, 19127)), ('Tim Kaine', (19118, 19127)), ('Jill Stein', (19469, 19479)), ('Gary Johnson', (19484, 19496))])]


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

    precision = tp / (tp + fp) if tp+fp > 0 else 0
    recall = tp / (tp + fn) if tp+fn > 0 else 0
    return precision, recall
    return f"Precision: {precision:.2f}\tRecall: {recall:.2f}"


def eval_funcs():
    log = []
    for client, model, method, chunksize in [
        # ('groq', 'llama-3.3-70b-versatile', 'embed', 1000),
        # ('groq', 'llama-3.3-70b-versatile', 'embed', 2000),
        # ('groq', 'llama-3.3-70b-versatile', 'embed', 5000),
        # ('groq', 'llama-3.3-70b-versatile', 'embed', 10000),
        # ('groq', 'llama-3.3-70b-versatile', 'embed', 20000),

        ('groq', 'qwen-qwq-32b', 'exact', 20000),
    ]:
        print(f"starting with {client=}, {model=}, {method=}, {chunksize=}")
        for prompt, filename, goldannos in dataset:
            print(f"doing {prompt}")
            # clear loggers
            l = logging.getLogger('functions')
            l.handlers.clear()
            l = logging.getLogger('output')
            l.handlers.clear()

            # create agent
            agent = Agent(mode='planner', client=client, model=model,
                          debug=False, test_classify=True, huggingGPT=True)
            agent.softwareenv.set_current_document_by_name(filename)
            methods = {
                "embed": agent.functionclass_classify.classify_span1,
                "embed_noICL": agent.functionclass_classify.classify_span_no_icl,
                "exact": agent.functionclass_classify.classify_span_exactwordcheck2,
            }

            # call function
            start_time = time.time()
            result = methods[method](criteria_query=prompt, scope='current document', chunk_size=chunksize)
            duration = time.time() - start_time
            precision, recall = compare(result, goldannos)

            found_words = [(agent.softwareenv.get_current_documenttext()[s:e], cat, s, e) for cat, (s, e) in result]

            log.append({
                "prompt": prompt,
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


if __name__ == "__main__":
    eval_funcs()
