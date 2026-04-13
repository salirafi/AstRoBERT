# AstRoBERT: An Abstract Recommender

AstRoBERT is a simple NLP-based recommender system built from the pre-trained [distilRoBERTa](https://huggingface.co/sentence-transformers/all-distilroberta-v1) transformer model and semantic search using [FAISS](https://faiss.ai), specifically for searching semantically similar abstracts for astrophysics papers uploaded to the [arXiv](https://arxiv.org/archive/astro-ph). The database is provided by the ArXiv team from [HERE](https://arxiv.org/archive/astro-ph) containing more than 381,000 papers as of March 2026.

🤗[ SEE THE DEMO IN MY HUGGINGFACE SPACE! ](https://huggingface.co/spaces/salirafi/AstRoBERT)🤗


Embeddings were generated for all papers in the dataset using a maximum sequence length of 512 tokens, the limit of the pre-trained model, producing 768-dimensional vectors. 

Evaluation was conducted using Precision@5, where relevance was defined by overlap in arXiv categories. This metric is somewhat biased because arXiv categories are broad and do not always reflect true semantic similarity. Compared with a TF-IDF baseline, the transformer-based embedding model achieved a Precision@5 of 0.7428, while TF-IDF achieved 0.7164, an improvement of 0.0264, which is somewhat marginal.

![/assets/image.png](/assets/image.png)

## Tools Used

### Backend

- pandas
- scikit-learn
- PyTorch
- transformers (for tokenizer)
- MySQL (cloud hosting for the live demo)
- Gradio

### Frontend
- HTML
- CSS

## Running

Most of the project's steps can be done by following the notebooks from `01_*.ipynb`, simple EDA and data extraction, to `04_*.ipynb`, the evaluation. Make sure to first run
```
pip install -r requirements.txt
```


## Sources

- [arXiv Metadata Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). Source of the raw paper metadata used in this project, including titles, abstracts, authors, categories, and update dates. It is updated weekly and is still going on as of the writing of this README.

- This project uses a pre-trained transformer model from [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1). Documentation about sentence-transformer in general can be found in [Sentence-Transformers Documentation](https://www.sbert.net/).

- [FAISS Documentation](https://faiss.ai/) is used for efficient similarity search over the many abstract embeddings.



## Author's Remarks

The use of generative AI includes: Github Copilot to help in code syntax and identifying bugs and errors. Outside of those, including problem formulation and framework of thinking, code logical reasoning and writing, from database management to web development, all is done mostly by the author.