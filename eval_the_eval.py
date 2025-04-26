import json
import os

from os import getcwd, listdir
from os.path import join

from pathlib import Path
if __name__ == "__main__":
    eval_folder = join(getcwd(), 'test_logs_end')
    # eval_file = "20250127-170918-oaa-ukp"
    # with open(join(eval_folder, f"{eval_file}.json")) as f:
    #     j = json.load(f)

    # total_tokens_prompt = 0
    # total_tokens_completion = 0
    # nb_runs = 0
    # for filetest in j:
    #     if not "method" in filetest:
    #         continue
    #     for run in filetest["tests"][0]["runs"]:
    #         if run["error"]:
    #             continue
    #         total_tokens_prompt += run["nb_tokens_prompt"]
    #         total_tokens_completion += run["nb_tokens_completion"]
    #         nb_runs += 1

    # total_tokens = total_tokens_prompt + total_tokens_completion
    # print(f"""{total_tokens_prompt}, {total_tokens_completion}\nTotal Tokens: {
    #       total_tokens}\nAvg Tokens: {total_tokens/nb_runs} """)

    # -------------------------------------
    # get scope

    # for filename in listdir(eval_folder):
    #     if filename.endswith("json"):
    #         with open(join(eval_folder, filename)) as f:
    #             d = json.load(f)
    #             nb_prompts_scope, cnt_scope = 0, 0
    #             nb_prompts_lf, cnt_lf = 0, 0
    #             for doc in d:
    #                 if not "tests" in doc:
    #                     continue

    #                 for run in doc["tests"][0]["runs"]:
    #                     if "get_scope" in run["expected"] and "get_scope" in run["executed"]:
    #                         nb_prompts_scope += 1
    #                         if run["correct_scope"] == True:
    #                             cnt_scope += 1
    #                     if "get_layer_and_feature" in run["expected"] and "get_layer_and_feature" in run["executed"]:
    #                         nb_prompts_lf += 1
    #                         if run["correct_layerandfeature"] == True:
    #                             cnt_lf += 1

    #             scoperate = round(cnt_scope/nb_prompts_scope, 2) if nb_prompts_scope > 0 else "-"
    #             lfrate = round(cnt_lf/nb_prompts_lf, 2) if nb_prompts_lf > 0 else "-"
    #             print(f"scope correct: {scoperate}\tlayerfeature correct: {lfrate}\t{filename=}")

    # -------------------------------------
    # get completely corrects

    for filename in sorted(Path(eval_folder).iterdir(), key=os.path.getmtime):
        if filename.name.endswith("json"):
            with filename.open() as f:
                d = json.load(f)
                nb_prompts, cnt = 0, 0
                for doc in d:
                    if not "tests" in doc:
                        continue
                    for run in doc["tests"][0]["runs"]:
                        error = run["error"]
                        lf = (run["correct_layerandfeature"] or run["correct_layerandfeature"] == "")
                        scope = (run["correct_scope"] or run["correct_scope"] == "")
                        valid = (run["is_valid"] or run["is_valid"] == "")

                        nb_prompts += 1
                        cnt += (valid and not error and lf and scope)
                print(f"{filename.stem}, completely correct: {cnt/nb_prompts}")
