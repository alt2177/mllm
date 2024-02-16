# Mixture of Large Language Models (MLLM): Blending Ensemble Learning and Deep Learning

Authors:
- Austin Tao ([austin.tao@berkeley.edu](mailto:austin.tao@berkeley.edu))
- Robert Thompson ([robert_thompson@berkeley.edu](mailto:robert_thompson@berkeley.edu))
- Phudish (Tam) Prateepamornkul ([phudish_p@berkeley.edu](mailto:phudish_p@berkeley.edu))
- Sean McAvoy ([sean_mcavoy@berkeley.edu](mailto:sean_mcavoy@berkeley.edu))

## Description

We aim to develop Mixture of Large Language Models (MLLM) that leverages several medium sized LLMs (~100 million parameters each) to attempt to match the performance of state-of-the-art LLMs such as GPT3.5 (~175 billion parameters at time of original publication). 

We specifically will be assessing summarization performance on a particular domain, in our case on the 
[Drug Review Dataset (Drugs.com) via UCI ML Repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com).

## Technologies

Our data is stored on [XetHub](https://about.xethub.com/?) at [https://xethub.com/alt2177/mllm-data](https://xethub.com/alt2177/mllm-data). We use both [Polars](https://github.com/pola-rs/polars) and [Pandas](https://pandas.pydata.org/) for our EDA and data pre-processing. For examples, see the `notebooks` folder.

##
to run the model_test > main.py
first the follow commands 
pip install transformers
pip install sentencepiece
then just run python main.py in the directory to answer questions