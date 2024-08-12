# Project: DSPy Optimization

This project utilizes the [DSPy](https://github.com/stanfordnlp/dspy)(Declarative Self-improving Language Programs) framework to algorithmically optimize LLM prompts. DSPy is capable of optimizing LLM weights and prompt system instructions, but here it will only be used to test if a few-shot prompt can improve RAG requests. The framework facilitates the use of test datasets to evaluate different LLM programs.    

The dataset used for this experiment consists of 64 question and answer examples about the 1965 Frank Herbert book Dune.

Like other LLM frameworks, DSPy provides utilities that ease the development of LLM programs. DSPy provides pre-canned strategies for prompting and LLM program optimization. It also facilitates evaluating LLM programs using test datasets.

In this experiment, the framework was used to provide evidence that RAG requests on the Dune corpus benefit from using few-shot prompting.


