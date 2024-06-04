from mlflow.metrics.genai import make_genai_metric_from_prompt, make_genai_metric, EvaluationExample

JUDGE_PROMPT = """
Task:
You must return the following fields in your response in two lines, one below the other:
score: Your numerical score for the model's faithfulness based on the rubric
justification: Your reasoning about the model's faithfulness score

You are an impartial judge. You will be given an input that was sent to a machine
learning model, and you will be given an output that the model produced. You
may also be given additional information that was used by the model to generate the output.

Your task is to determine a numerical score called faithfulness based on the input and output.
A definition of faithfulness and a grading rubric are provided below.
You must use the grading rubric to determine your score. You must also justify your score.

Examples could be included below for reference. Make sure to use them as references and to
understand them before completing the task.

Input:
{input}

Output:
{output}

{grading_context_columns}

Metric definition:
Faithfulness is only evaluated with the provided output and provided context, please ignore the provided input entirely when scoring faithfulness. Faithfulness assesses how much of the provided output is factually consistent with the provided context. A higher score indicates that a higher proportion of claims present in the output can be derived from the provided context. Faithfulness does not consider how much extra information from the context is not present in the output.

Grading rubric:
Faithfulness: Below are the details for different scores:
- Score 1: None of the claims in the output can be inferred from the provided context.
- Score 2: Some of the claims in the output can be inferred from the provided context, but the majority of the output is missing from, inconsistent with, or contradictory to the provided context.
- Score 3: Half or more of the claims in the output can be inferred from the provided context.
- Score 4: Most of the claims in the output can be inferred from the provided context, with very little information that is not directly supported by the provided context.
- Score 5: All of the claims in the output are directly supported by the provided context, demonstrating high faithfulness to the provided context.

Examples:

Example Output:
mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. 

Additional information used by the model:
key: context
value:
mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function.

Example score: 2
Example justification: The output provides a working solution, using the mlflow.autolog() function that is provided in the context.
        

Example Output:
mlflow.autolog(disable=True) will disable autologging for all functions.

Additional information used by the model:
key: context
value:
mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function.

Example score: 5
Example justification: The output provides a solution that is using the mlflow.autolog() function that is provided in the context.
        

You must return the following fields in your response in two lines, one below the other:
score: Your numerical score for the model's faithfulness based on the rubric
justification: Your reasoning about the model's faithfulness score

Do not add additional new lines. Do not add any other fields.
"""

context_precision = make_genai_metric_from_prompt(
    name="context_precision",
    judge_prompt=JUDGE_PROMPT,
    model="openai:/gpt-3.5-turbo-16k",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)


definition = """
Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are 
ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the 
question, ground_truth and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.
"""

grading_prompt = """
Step 1: For each chunk in retrieved context, check if it is relevant or not relevant to arrive at the ground truth for the given question.
Step 2: Calculate precision@k for each chunk in the context.
Precision@1 = 0/1 = 0
Precision@2 = 1/2 = 0.5
Step 3: Calculate the mean of precision@k to arrive at the final context precision score.
Context Precision= (0+0.5)/1 = 0.5
"""

examples = [
    EvaluationExample(
        input='When was the first super bowl?',
        output='The first superbowl was held on Jan 15, 1967',
        score=1,
        justification="",
        grading_context={
            "ground_truth": 'The first superbowl was held on January 15, 1967',
            "context": ['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,']
        },
    ),
    EvaluationExample(
        input='When was the first super bowl?',
        output='The most super bowls have been won by The New England Patriots',
        score=0.5,
        justification="",
        grading_context={
            "ground_truth": 'The New England Patriots have won the Super Bowl a record six times',
            "context": ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']
        },
    )
]

context_precision = make_genai_metric(
        name="context_precision",
        definition=definition,
        grading_prompt=grading_prompt,
        include_input=False,
        examples=examples,
        version="v1",
        model="openai:/gpt-3.5-turbo-16k",
        grading_context_columns=["ground_truth", "context"],
        parameters={"temperature": 0.0},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )

if __name__ == '__main__':
    print(context_precision)
    pass