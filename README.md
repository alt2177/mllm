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
to attempt to match the performance of larger, state-of-the-art LLMs. We focus on classification tasks and

classification performance on a particular domain, in our case on the
[Drug Review Dataset (Drugs.com) via UCI ML Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com).

## Technologies

We will be using [MergeKit](https://github.com/arcee-ai/mergekit) to run our LLM merging operations. In preliminary tests there are several different types we need to test and work through to see what our best options are.

So far we are evaluating [FLAN](https://huggingface.co/docs/transformers/model_doc/flan-t5) pretrained models on Huggingface, and are considering merging with smaller iterations of [LLama](https://huggingface.co/meta-llama) and some other more specific models. We will be evaluating performance of merged models in the coming weeks.

## Progress

We include in this repo notebooks for EDA. We have worked with our data to clean and pre process to make sure we are ready for model evaluation. We have also extracted relevant summary statistics regarding our text fields, and we are planning on incorporating these summary stats into our evaluation metrics.

We have also gotten our preliminary LLMs running and generating output, however they are currently not present in this repository. The reason for this is because of their size. To download and operate them we are currently using the SCF cluster, and cannot simply drop the LLMs into a notebook and run them in this repository. We will try to think of more creative solutions for showing model progress in the future.

## Timeline

Now that we have gotten models running on the SCF, our data cleaned and summary stats extracted, we are currently working on the merging aspect of our project. Over the next couple weeks we will be working to test out mergekit and evaluate performance improvements (or degredations) that get brought about by our merging techniques. After that, we hope to work on exploring alternate techniques for merging, larger amounts of models combine (Mixture of Experts), and comparing performance against higher parameter LLMs.

## Challenge

The biggest continued challenge remains metric quantification. It is not always clear how we should be measuring our output, and what we should benchmark against. We should put extra thought into this over the next 2 weeks while we start merging and testing our LLMs so we have a plan to evaluate.
