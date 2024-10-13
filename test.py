import logging.config
import unittest
import csv
import os
import logging
import tqdm
import sys

from os import getcwd
from os.path import join
from parser import analyze_functions
from agent import call_llm_planner
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class TestParser(unittest.TestCase):
    def setUp(self):
        # load groq
        self.client = Groq(
            api_key=GROQ_API_KEY,
        )

    def test_expected_functions(self):
        models = ["llama3-70b-8192"]
        files = ["test_expectations_Cheetah.csv",
                 "test_expectations_Politician.csv"]

        for file in files:
            for model in models:
                self.helper_test_planner(file, model)

    def helper_test_planner(self, file, model):
        logger.info("testing %s with %s", file, model)
        with open(join(getcwd(), "testfiles", file), newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            correct_results = 0
            for i, row in enumerate(reader):
                prompt, expected_results, scope, layer, feature, label, criteria = row
                llm_response = call_llm_planner(self.client, model, prompt)
                detected = set([func for _, func,
                                _ in analyze_functions(llm_response)])
                expected_results = set(expected_results.split(","))
                correct_result = detected == expected_results
                correct_results += correct_result
                logger.debug("Analyzing prompt: %s", prompt)
                logger.debug("Expected functioncalls: %s", expected_results)
                logger.debug("Detected functioncalls: %s", detected)
                logger.debug("Correct Result?: %s\n", correct_result)
        logger.info(
            "%d/%d functions were correctly called from the planner.", correct_results, i+1)


if __name__ == "__main__":
    logger = logging.getLogger("tests")
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout)
    unittest.main()
