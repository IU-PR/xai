# Retrieval-augmented generation

Project by:
- Vladislav Urzhumov
- Danil Timofeev

## What is RAG?
Retrieval-Augmented Generation is a framework in the explainable artificial intelligence field for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information. 
By grounding an LLM on a set of external, verifiable facts, the model has fewer opportunities to pull information baked into its parameters. This reduces the chances that an LLM will leak sensitive data, or "hallucinate" incorrect or misleading information. 

## Advantages of RAG

* _Transparency_: generated response is supported by the document chunks model referred to. A short and easy-to-go variant of references.
* _Better accountability_: RAG allows to trace back the information source to understand, whether the mistake in the response was due to incorrect source data or a flaw in the model’s processing
* _Hallucinations avoided_: knowledge can be directly revised and expanded, and accessed knowledge can be inspected and interpreted.
* _Reasonably limited documents_: users can limit the external resources to provide responses based only one the trusted source, enhanced by the pre-trained model’s ability to construct adequate connected sentences.
* _Ease of implementation_: modern frameworks allow embracing of Retrieval-Augmented Generation with ease and small adjustments of code. One can always add improvements on top to achieve better results, however, standard RAG is easy-to-go.
* _Model-agnostic explainability of LMs_: any LLM's responses can be explained via RAG, despite of the model type and number of parameters.
* _Responses based on trusted documents_: there is an ability to construct a library of trusted documents to use while generating results, avoiding distrusted sources.

## Technical steps of RAG

Documents, which are provided for LLM to take information from, are firstly properly indexed.
In vanilla (naive) RAG, user query is not pre-processed specifically, thus validated user query plus the indexed documents are sent to the retrieval stage.

Relevant chunks of information are retrieved, included to prompt and then processed by LLM. Output is provided coupled with the indexed retrieved chunks of information.

## RAG improvements

Retrieval-Augmented Generation, as a popular framework, was highly researched. Thus, numerous improvement techniques are present to enhance the performance of the technique.

Before the retrieval part, Pre-retrieval is recommended (such as query routing or query expansion) to provide more relevant or sufficient context.

After the retrieval part, Post-retrieval is used to summarize, fuse or rerank retrieved results.

Our framework research and implementation embrace the pre-retrieval addition, leaving the post-retrieval for the readers to try out and make an experiment.

## Implementation details

- LLM used is `llama-3-8b-8192` from Groq. We picked the Groq provider for open-source models because it’s the quickest one out there and llama-3-8b is the current state-of-the-art for small language models.
- Embedding model embraced is `bge-base-en-v1.5`. The embedding model used is also open-source, it’s currently the SOTA for its current size. We could have used a bigger more capable multilingual model, but we decided to just keep the English support, so this was sufficient.
- RAG framework for our purpose should be well-maintained and easy-to-use. Thus, we have chosen `llama-index` (vector databases and retrieval);
- As an improvement, our team has used AutoContext to pre-retrieve the necessary information. We have used it to correctly summarize the main idea behind the whole document with specific details in separate chunks.

## Code

Best way to check code is interactive environment, such as Google Colab.
Hence, our team provides one for any developer interested in trying by own hand.

Please, enjoy by following the [link](https://colab.research.google.com/drive/1chU3jbysPW3z9j6b8zrg--P9UUXkrfKq?usp=sharing).
Don't forget to insert your own Groq API key here (it's free):
```py
if "llm" not in st.session_state:
    st.session_state.llm = Groq(
        model="llama3-8b-8192",
        api_key="",
        kwargs={""},
    )
    Settings.llm = st.session_state.llm
```

## Further improvements

* _Better pipelines_: Retrieval can be improved by implementing more complex RAG pipelines, adding better data parsing (better chunking, for example) and adding a reranker, for example (a step in post-retrieval);
* _Multilingual support_: users are interested in explained results not only in english, but also in other languages;
* _OCR support_: Optical Character Recognition (OCR) is a technology that falls under the umbrella of machine learning and computer vision. It involves converting images containing written text into machine-readable text data. Being able to process images with infographics or screenshots along with non-copiable image-based pdfs is a big step towards versatility;
* _Specific rules for scientific paper_: Pre-retrieval for scientific paper should be focused on abstracts, summaries and conclusions written by human authors rather than generated ones, that will increase understanding and trustworthiness of documents and retrieval pipeline.

### Thanks for the attention!
