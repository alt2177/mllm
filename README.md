# Merge of Large Language Models (MLLM): Blending Ensemble Learning and Deep Learning

Authors:

- Austin Tao ([austin.tao@berkeley.edu](mailto:austin.tao@berkeley.edu))
- Robert Thompson ([robert_thompson@berkeley.edu](mailto:robert_thompson@berkeley.edu))
- Phudish (Tam) Prateepamornkul ([phudish_p@berkeley.edu](mailto:phudish_p@berkeley.edu))
- Sean McAvoy ([sean_mcavoy@berkeley.edu](mailto:sean_mcavoy@berkeley.edu))

## Description

Merging models has come about as a popular new paradigm for improving pretrained model
performance at lower paramaters and without the need to retrain. There are several ways
to merge models and a significant amount of work right now going into testing
different combinations of models to find out what can bring about large improvements.

We aim to benchmark and evaluate a Merge of Large Language Models (MLLM) that leverages several medium sized LLMs (~100 million parameters each)
to attempt to match the performance of larger, state-of-the-art LLMs. We focus on classification tasks and classification performance on a particular domain, in our case on the
[Drug Review Dataset (Drugs.com) via UCI ML Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com).

## Technologies

We use [MergeKit](https://github.com/arcee-ai/mergekit) to run our LLM merging operations. Specifically,
we evalaute three merge techniques:

- Linear Merge [Wortsman et al.](https://doi.org/10.48550/arXiv.2203.05482)
- TIES Merge [Yadav et al.](https://doi.org/10.48550/arXiv.2306.01708)
- DARE + TIES/Linear Merge [Yu et al.](https://doi.org/10.48550/arXiv.2311.03099)

For the base LLM, we use GPT-2 [Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) for
its relatively small size and moderate base performance.

## Running Our Experiments

[TODO]
