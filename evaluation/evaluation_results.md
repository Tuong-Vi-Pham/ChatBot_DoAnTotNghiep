# RAG System Evaluation Metrics Report

Evaluation results for retrieval, generation, and intent clarification modules.

## 1. Retrieval Evaluation Results

| Metric | Value |
| :--- | :--- |
| Precision@1 | 0.6250 |
| Precision@3 | 0.2292 |
| Precision@5 | 0.1375 |
| Recall@1 | 0.6250 |
| Recall@3 | 0.6875 |
| Recall@5 | 0.6875 |
| F1-score@3 | 0.3438 |
| MAP | 0.6458 |
| MRR | 0.6458 |
| NDCG@3 | 0.6562 |
| NDCG@5 | 0.6562 |
| NMAE | 0.4480 |

## 2. Generation Evaluation Results

| Metric | Value |
| :--- | :--- |
| ROUGE-1 | 0.3195 |
| ROUGE-2 | 0.2044 |
| ROUGE-L | 0.2813 |
| BLEU-1 | 0.1773 |
| BLEU-2 | 0.1321 |
| BLEU-4 | 0.0559 |
| METEOR | 0.4505 |
| BERTScore Precision | 0.7513 |
| BERTScore Recall | 0.8983 |
| BERTScore F1 | 0.8164 |
| LLM Judge Accuracy (1-5) | 4.5000 |
| LLM Judge Relevancy (0-1) | 0.8750 |
| LLM Judge Faithfulness (0-1) | 0.8750 |

## 3. Intent Clarification Evaluation Results

| Metric | Value |
| :--- | :--- |
| Clarification Success Rate (CSR) | 1.0000 |
| Average Clarification Turn Count | 2.0000 |
| Post-Clarification Accuracy | 0.3333 |
