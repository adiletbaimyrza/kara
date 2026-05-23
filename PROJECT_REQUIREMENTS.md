
# Final Project — NLP (Master's Program)

# TL;DR — Project Rules in 10 Sentences

1. The project is **individual** and covers any problem in the field of NLP.

2. You may use your own data or public datasets (e.g., Kaggle, Hugging Face, scraping).

3. The project must include a **simple baseline** as a reference point.

4. You are required to compare **at least 3 experiment variants** (e.g., preprocessing, model, prompt, retriever).

5. The most important thing is to show **how design decisions affect the result**.

6. Include a meaningful evaluation: a results table, a chart, and a brief discussion.

7. Show several examples of model errors and try to explain where they come from.

8. At the end, write short conclusions: what helped, what didn't work, and what you learned.

9. You may use pre-built models and LLMs, but **there is no need to and we do not recommend using paid APIs** — prefer free and reproducible solutions.

10. You submit a working notebook with code, and the grade is based mainly on **experiments, analysis, and conclusions — not on the highest score alone**.


# Details

## What is this about?

The task is to prepare an **individual NLP project on any topic**.

The most important thing in the project is not "getting the best score", but demonstrating that you understand:

- how data affects the model,

- how preprocessing changes results,

- when a more complex model helps,

- how to analyze errors,

- how to draw conclusions from experiments.


---

## Project Topic

The topic is open, as long as it relates to NLP. The topic must fit one of the following tracks:

### Track A — Classical Supervised NLP

- text classification

- sentiment analysis

- spam / fake news

- intent classification

- authorship / style detection

- NLI / stance detection


### Track B — Information Extraction

- NER

- slot filling

- keyword extraction

- relation extraction

- entity linking


### Track C — Retrieval and Embeddings

- semantic search

- duplicate question detection

- document clustering

- retrieval ranking

- recommendation over text


### Track D — Generative NLP / LLM

- summarization

- QA

- RAG

- task-oriented chatbot

- prompt engineering

- fine-tuning a small model

- comparison of foundation models

## Data

You may use:

- public datasets (e.g., Kaggle, HuggingFace Datasets, UCI, your own scraping)

- your own data

- data from APIs or web services


The project must clearly describe:

- the data source

- the number of examples

- how the data was cleaned

- the train / validation / test split

- potential data limitations (noise, bias, class imbalance)


You may use your own data or public datasets.

---

## What must the project include?

### 1) Baseline

First, prepare a **simple baseline solution**, e.g.:

- TF-IDF + Logistic Regression

- embedding search

- zero-shot prompting

- a simple pretrained model


This is the reference point for further experiments.

---

### 2) At Least 3 Experiments

This is the most important part of the project.

Show **at least 3 solution variants** that answer a specific question.

For example:

- does preprocessing help?

- does lemmatization improve the result?

- does a transformer give better results?

- does a better prompt work better?

- how does chunking affect RAG?


The goal is to show that the project was thought through.

---

### 3) Evaluation

Compare results using meaningful metrics, e.g.:

- accuracy / F1

- ROUGE / BLEU

- Recall@K / MRR

- human evaluation


Recommended:

- results table

- chart

- brief discussion


---

### 4) Error Analysis

Present:

- what works well,

- where the model makes mistakes,

- what the typical errors are,

- what might be causing them.


This is often more important than the metric itself.

---

### 5) Conclusions

At the end, briefly cover:

- what helped the most,

- what didn't work,

- what surprised you,

- what you learned.


---

## Can I use LLMs/APIs?

Yes — you may use pre-built models, Hugging Face models, and LLM APIs.

But:

> **The LLM should be a tool, not the entire solution.**

The project cannot rely solely on:

- a single API call,

- a ready-made notebook from the internet,

- a tutorial without your own experiments.


The most important things are **comparisons and conclusions**.

---

## What do you need to prepare for the defense?

- a `.ipynb` notebook

- the code needed to run it

- a short description at the beginning of the notebook

- required libraries


The notebook should run from start to finish.

---

## How will the project be graded?

You will earn the most points for:

- a meaningful problem and data

- a good baseline

- interesting experiments

- showing the impact of preprocessing / model choices

- error analysis

- meaningful conclusions

- a readable notebook


---

## The Most Important Thing

I am not interested in a project like:

> "I used a large model and it works."

I am interested in a project like:

> "I tested 3 approaches, preprocessing improved the result by 5%, but using a larger model didn't help further."

That is what a **good research-and-experiment project** looks like.


## Grading Criteria (100 pts)

| Element | Points |
| --- | --- |
| Problem definition and data | 20 |
| Baseline and experiment methodology | 35 |
| Model / pipeline implementation | 20 |
| Results and error analysis | 15 |
| Code quality and reproducibility | 10 |
| If the project is in Polish | 10 (bonus) |


# 10 Example Projects

| # | Project | Difficulty (1–5) | Data Issues? |
| --- | --- | --- | --- |
| 1 | Sentiment analysis of product reviews | 1 | **No** – very easy (Amazon, IMDB, Allegro, Kaggle) |
| 2 | Fake news / clickbait detection | 2 | **Minor** – public data is available but can be noisy |
| 3 | NER for job postings (skills, positions, technologies) | 3 | **Medium** – data easy to collect, labels harder |
| 4 | Duplicate question detection (e.g., StackOverflow/Quora) | 2 | **No** – very good public datasets |
| 5 | Semantic search over scientific articles or PDF documents | 3 | **No** – data easy from arXiv / your own PDFs |
| 6 | RAG chatbot answering questions from course documentation | 4 | **No** – you can use your own course materials |
| 7 | Automatic summarization of news articles | 4 | **Minor** – many datasets, but evaluation is harder |
| 8 | Intent classification of emails / support tickets | 3 | **Medium** – data needs to be found or built |
| 9 | Prompt engineering vs fine-tuning a small model | 5 | **No** – ready-made datasets from HF can be used |
| 10 | Hallucination detection in RAG system responses | 5 | **Yes** – hardest; data often needs to be prepared manually |


# FAQ — NLP Project

## Can I use ChatGPT, LLM models, and ready-made APIs?

Yes, but **it is not required**.

You may use:

- models from Hugging Face

- local open-source models

- pre-built embeddings

- free APIs

- your own models trained from scratch or fine-tuned


**There is no requirement to use paid APIs** (e.g., OpenAI, Anthropic, Google).

We in fact **discourage choosing projects that require financial costs**, because:

- not everyone has the same budget,

- it makes the project harder to reproduce,

- it limits the ability to re-run the notebook,

- it often makes fair experiment comparisons harder.


The most recommended solutions are:

- local

- open-source

- reproducible without additional costs


---

## Can I do a RAG-based project or a chatbot?

Yes.

It is a very good project topic, as long as you:

- compare several retrievers or embeddings,

- check the effect of chunking,

- test different prompts,

- show typical hallucinations and errors,

- analyze the effect of pipeline parameters on the result


A chatbot alone without experiments is not enough.

---

## Do I need to train my own model?

No.

You may:

- use a pretrained model

- do fine-tuning

- use sentence embeddings

- use a classical model

- use a retrieval-based approach

- use zero-shot prompting


The most important thing is **comparing variants and drawing conclusions**.

---

## Can I use a ready-made dataset?

Yes, and it is usually the best choice.

You may use:

- Kaggle

- Hugging Face Datasets

- UCI

- public benchmarks

- your own data

- data from your own scraping


Just remember to:

- describe the source,

- justify the choice,

- check data quality,

- show preprocessing.


---

## Can I use a ready-made notebook from the internet?

You may use it for inspiration, but:

> a ready-made tutorial run without your own modifications **is not a project**.

The project must include:

- your own research question,

- a baseline,

- at least 3 experiments,

- error analysis,

- your own conclusions.


---

## Is the metric score the most important thing?

No.

Much more important is:

- a meaningful research question,

- a good baseline,

- comparison of variants,

- error analysis,

- the ability to explain **why the result changed**


A project with F1 = 0.78 and great analysis is better than a project with F1 = 0.90 with no conclusions.

---

## Can I do text classification?

Yes, of course.

It is still a very good topic, but try to go beyond:

> "I took a dataset from Kaggle and ran BERT."

For example, compare:

- TF-IDF vs embeddings

- with preprocessing vs without preprocessing

- classical model vs transformer

- the effect of augmentation

- class balancing


---

## Can I do a project on my own data?

Yes, and we highly recommend it.

Your own data often leads to the most interesting projects, because it is easier to see:

- data quality problems

- noise

- annotation errors

- model limitations


This is great material for error analysis.

---

## What if the data turns out to be poor?

That **is also a valuable project result**.

You can then show:

- how data quality affects the model,

- which classes are problematic,

- how preprocessing helps or hurts,

- what data would be worth collecting better


In NLP, the problem is very often not the model, but the data.
