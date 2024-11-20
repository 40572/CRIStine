import nest_asyncio
nest_asyncio.apply()
import os
import openai
from getpass import getpass


openai.api_key = "sk-w7KwcyTsWL-l5waK0w6OTtJUCzyeukZKL20tqjZOCGT3BlbkFJpKzJDQ0qVEZ53ip3G_TBBAq-Ed08kpDZgxb9t6P60A"

os.environ["OPENAI_API_KEY"] = "sk-w7KwcyTsWL-l5waK0w6OTtJUCzyeukZKL20tqjZOCGT3BlbkFJpKzJDQ0qVEZ53ip3G_TBBAq-Ed08kpDZgxb9t6P60A"

from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import SummarizationMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import ToxicityMetric

def set_key():
    OPENAI_API_KEY = "sk-w7KwcyTsWL-l5waK0w6OTtJUCzyeukZKL20tqjZOCGT3BlbkFJpKzJDQ0qVEZ53ip3G_TBBAq-Ed08kpDZgxb9t6P60A"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def deep_eval_bias(user_input, response):
    metric = BiasMetric(threshold=0.5)
    test_case = LLMTestCase(
        input=user_input,
        actual_output=response
    )
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results

def deep_eval_correctness(user_input, response,doc_retriever):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the documents supplied.",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also  penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    documents_ret = doc_retriever.query(user_input)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=response,
        expected_output=documents_ret
    )

    correctness_metric.measure(test_case)
    eval_results =[correctness_metric.score, correctness_metric.reason]
    return eval_results

def deep_eval_summary(user_input, response):
    test_case = LLMTestCase(input=input, actual_output=response)
    metric = SummarizationMetric(
        threshold=0.5,
        model="gpt-4",
        assessment_questions=[
            "Is the coverage score based on a percentage of 'yes' answers?",
            "Does the score ensure the summary's accuracy with the source?",
            "Does a higher score mean a more comprehensive summary?"
        ]
    )
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results

def deep_eval_relevancy(user_input, response):
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4",
        include_reason=True
    )
    test_case = LLMTestCase(
        input=user_input,
        actual_output=response
    )
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results


def deep_eval_faithfulness(user_input, response,doc_retriever):
    metric = FaithfulnessMetric(
        threshold=0.5,
        model="gpt-4",
        include_reason=True
    )
    documents_ret = [doc_retriever.query(user_input)]
    test_case = LLMTestCase(
        input=user_input,
        actual_output=response,
        retrieval_context=[str(documents_ret)]
    )
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results

def deep_eval_hallucination(user_input, response,doc_retriever):
    documents_ret = [doc_retriever.query(user_input)]
    test_case = LLMTestCase(
        input=user_input,
        actual_output=response,
        context=[str(documents_ret)]
    )
    metric = HallucinationMetric(threshold=0.5)
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results

def deep_eval_toxicity(user_input, response):
    metric = ToxicityMetric(threshold=0.5)
    test_case = LLMTestCase(
        input=user_input,
        # Replace this with the actual output from your LLM application
        actual_output=response
    )
    metric.measure(test_case)
    eval_results =[metric.score, metric.reason]
    return eval_results



    












