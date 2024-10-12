import unittest
import csv
import os

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
        self.client = Groq(
            api_key=GROQ_API_KEY,
        )

    def test_parser_cheetah(self):
        with open(join(getcwd(), "testfiles", "test_expectations_Cheetah.csv"), newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                prompt, expected_results, scope, layer, feature, label, criteria = row
                llm_response = call_llm_planner(self.client, prompt)
                print(f"\n{prompt}\n\n{llm_response}\n")


if __name__ == "__main__":
    unittest.main()
