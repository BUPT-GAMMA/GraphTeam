# NLGraph Evaluation Guide

Welcome to the evaluation guide for NLGraph. In this guide, we will walk you through the steps needed to evaluate the results generated by the large language model.

## Steps to Evaluate

1. Generate Results: Run the large language model on the `NLGraph_test.csv` dataset and generate the results. Make sure to save these results for the next step.

2. Add Results to Dataset: Open the `NLGraph_test.csv` file and create a new column named `result`. Paste the results generated in the previous step into this new column. Make sure that each result aligns correctly with its corresponding input data in the dataset.

3. Run Evaluation: Navigate to the directory containing the `evaluate.py` script and run it. This script will evaluate the results in the `result` column of the `NLGraph_test.csv` dataset.

Here is the command to run the evaluation script:

```bash
python evaluate.py
```

Please ensure that you have the necessary permissions to run this script and that your Python environment has all the necessary dependencies installed.

That's it! The `evaluate.py` script will now evaluate your results and provide you with an evaluation score.