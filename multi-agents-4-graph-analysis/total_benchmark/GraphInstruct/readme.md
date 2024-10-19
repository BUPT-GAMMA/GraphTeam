# GraphInstruct Benchmark

## Introduction
This project is based on [GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability](https://arxiv.org/pdf/2403.04483 https://github.com/CGCL-codes/GraphInstruct). We have regenerated the data according to the methods proposed in the paper and cleaned the data in terms of format and content. The cleaned data is placed in the `modify` folder, which retains the original data, while the cleaned data is saved in CSV files starting with `checked_modified`.

## Evaluation Method
The evaluation was performed using GPT 40-mini, which can be invoked through the content in `ask.ipynb`. The evaluation results are further processed in `check.ipynb`. For detailed inquiry methods and evaluation procedures, refer to both notebook files.