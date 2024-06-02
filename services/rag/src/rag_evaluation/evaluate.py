import mlflow
import pandas as pd

from src.chains import get_in_memory_retriever_qa_chain
from src.config import HERCULES_VALIDATION_SET
from src.rag_evaluation.faithfulness_examples import faithfulness_metric, relevance_metric

if __name__ == '__main__':
    eval_df = pd.read_csv(
        str(HERCULES_VALIDATION_SET),
        usecols=['Question', 'Ground Truth'],
    )
    chain, vectorstore = get_in_memory_retriever_qa_chain()

    def model(input_df):
        answer = []
        for index, row in input_df.iterrows():
            answer.append(chain.invoke(row["Question"]))

        return answer

    # system_prompt = "Answer the following question in two sentences"
    # # Wrap "gpt-4" as an MLflow model.
    # logged_model_info = mlflow.openai.log_model(
    #     model="gpt-4",
    #     task=openai.chat.completions,
    #     artifact_path="model",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": "{question}"},
    #     ],
    # )

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow LLM Evaluation")

    # Start an MLflow run
    with mlflow.start_run():
        # # Log the hyperparameters
        # mlflow.log_params(params)
        #
        # # Log the loss metric
        # mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("LLM Evaluation", "quick-demo-1")

        results = mlflow.evaluate(
            model,
            eval_df,
            model_type="question-answering",
            evaluators="default",
            predictions="result",
            extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
            evaluator_config={
                "col_mapping": {
                    "inputs": "Question",
                    "context": "source_documents",
                }
            },
        )
        print(results.metrics)

    # with mlflow.start_run():
    #     # Log the LangChain LLMChain in an MLflow run
    #     logged_model = mlflow.langchain.log_model(
    #         chain,
    #         "langchain_model",
    #     )
    #
    #     # Load the logged model using MLflow's Python function flavor
    #     loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    #
    #     # Use predefined question-answering metrics to evaluate our model.
    #     results = mlflow.evaluate(
    #         loaded_model,
    #         eval_data,
    #         targets="Ground Truth",
    #         model_type="question-answering",
    #     )
    #     print(f"See aggregated evaluation results below: \n{results.metrics}")
    #
    #     # Evaluation result for each data record is available in `results.tables`.
    #     eval_table = results.tables["eval_results_table"]
    #     print(f"See evaluation table below: \n{eval_table}")
    #
    #     vectorstore.delete_collection()
