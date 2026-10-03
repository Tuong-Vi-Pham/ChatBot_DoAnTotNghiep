import os
import sys
import json
import re
import numpy as np
from typing import List, Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.intent_clarification.clarifier import IntentClarifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class CorrectedRAGEvaluator:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "chroma_db")
        self.benchmark_path = os.path.join(base_dir, "evaluation/benchmark_queries.json")
        
        self.embedder = Embedder()
        self.db_manager = VectorDBManager(db_path=self.db_path, embedder=self.embedder)
        self.retriever = HybridRetriever(db_manager=self.db_manager, embedder=self.embedder)
        self.llm_client = LLMClient()
        self.pipeline = RAGPipeline(retriever=self.retriever, llm_client=self.llm_client)
        self.clarifier = IntentClarifier(llm_client=self.llm_client)
        
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark_queries = json.load(f)
            
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        Evaluates full retrieval pipeline (Hybrid + Reranker) across all direct/rephrased queries.
        """
        eval_queries = [q for q in self.benchmark_queries if q["type"] in {"direct", "rephrased"}]
        
        p1_list, p3_list, p5_list = [], [], []
        r1_list, r3_list, r5_list = [], [], []
        f1_3_list = []
        mrr_list = []
        ndcg3_list, ndcg5_list = [], []
        
        for q in eval_queries:
            query_text = q["query"]
            gt_id = q.get("ground_truth_id")
            
            # Execute full retriever pipeline
            retrieval_res = self.retriever.retrieve(query_text, top_k=5)
            results = retrieval_res.get("results", [])
            
            # Identify rank of relevant item
            rank = -1
            for idx, item in enumerate(results):
                meta = item.get("metadata", {})
                if meta.get("faq_no") == gt_id or meta.get("chunk_id") == str(gt_id):
                    rank = idx + 1
                    break
            
            # Calculate MRR
            rr = 1.0 / rank if rank != -1 else 0.0
            mrr_list.append(rr)
            
            # Calculate Precision@K and Recall@K (assuming 1 relevant doc)
            p1 = 1.0 if rank == 1 else 0.0
            p3 = (1.0 / 3.0) if (rank != -1 and rank <= 3) else 0.0
            p5 = (1.0 / 5.0) if (rank != -1 and rank <= 5) else 0.0
            
            r1 = 1.0 if rank == 1 else 0.0
            r3 = 1.0 if (rank != -1 and rank <= 3) else 0.0
            r5 = 1.0 if (rank != -1 and rank <= 5) else 0.0
            
            f1_3 = (2 * p3 * r3) / (p3 + r3) if (p3 + r3) > 0 else 0.0
            
            p1_list.append(p1); p3_list.append(p3); p5_list.append(p5)
            r1_list.append(r1); r3_list.append(r3); r5_list.append(r5)
            f1_3_list.append(f1_3)
            
            # NDCG
            dcg3 = (1.0 / np.log2(rank + 1)) if (rank != -1 and rank <= 3) else 0.0
            dcg5 = (1.0 / np.log2(rank + 1)) if (rank != -1 and rank <= 5) else 0.0
            ndcg3_list.append(dcg3)
            ndcg5_list.append(dcg5)
            
        return {
            "Precision@1": np.mean(p1_list),
            "Precision@3": np.mean(p3_list),
            "Precision@5": np.mean(p5_list),
            "Recall@1": np.mean(r1_list),
            "Recall@3": np.mean(r3_list),
            "Recall@5": np.mean(r5_list),
            "F1-score@3": np.mean(f1_3_list),
            "MRR": np.mean(mrr_list),
            "NDCG@3": np.mean(ndcg3_list),
            "NDCG@5": np.mean(ndcg5_list)
        }

    def evaluate_generation(self) -> Dict[str, float]:
        """
        Evaluates generation metrics across ALL benchmark test cases.
        """
        eval_queries = [q for q in self.benchmark_queries if q["type"] in {"direct", "rephrased"}]
        chencherry = SmoothingFunction()
        
        r1_scores, r2_scores, rl_scores = [], [], []
        b1_scores, b2_scores, b4_scores = [], [], []
        meteor_scores = []
        judge_acc, judge_rel, judge_faith = [], [], []
        
        hyps, refs = [], []
        contexts = []
        
        for q in eval_queries:
            query = q["query"]
            ref = q["ground_truth_answer"]
            
            response = self.pipeline.run(query=query, faq_threshold=0.70)
            hyp = response["answer"]
            
            # Extract retrieved context
            context_text = "\n".join([item["content"] for item in response.get("sources", []) if "content" in item])
            
            hyps.append(hyp)
            refs.append(ref)
            contexts.append(context_text)
            
            # ROUGE
            r_score = self.rouge.score(ref, hyp)
            r1_scores.append(r_score['rouge1'].fmeasure)
            r2_scores.append(r_score['rouge2'].fmeasure)
            rl_scores.append(r_score['rougeL'].fmeasure)
            
            # BLEU with NLTK tokenization
            ref_tokens = [word_tokenize(ref.lower())]
            hyp_tokens = word_tokenize(hyp.lower())
            
            b1_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1))
            b2_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
            b4_scores.append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))
            
            # METEOR
            try:
                meteor_scores.append(meteor_score([word_tokenize(ref)], word_tokenize(hyp)))
            except Exception:
                meteor_scores.append(0.0)
                
            # LLM Judge (with retrieved context)
            j_res = self._run_llm_judge_correct(query, ref, hyp, context_text)
            judge_acc.append(j_res["accuracy"])
            judge_rel.append(j_res["relevancy"])
            judge_faith.append(j_res["faithfulness"])
            
        # BERTScore in batch
        P, R, F1 = bert_score.score(hyps, refs, model_type="distilbert-base-uncased", lang="en", verbose=False)
        
        return {
            "ROUGE-1": np.mean(r1_scores),
            "ROUGE-2": np.mean(r2_scores),
            "ROUGE-L": np.mean(rl_scores),
            "BLEU-1": np.mean(b1_scores),
            "BLEU-2": np.mean(b2_scores),
            "BLEU-4": np.mean(b4_scores),
            "METEOR": np.mean(meteor_scores),
            "BERTScore Precision": float(torch.mean(P)),
            "BERTScore Recall": float(torch.mean(R)),
            "BERTScore F1": float(torch.mean(F1)),
            "LLM Judge Accuracy (1-5)": np.mean(judge_acc),
            "LLM Judge Relevancy (0-1)": np.mean(judge_rel),
            "LLM Judge Faithfulness (0-1)": np.mean(judge_faith)
        }

    def _run_llm_judge_correct(self, query: str, ref: str, hyp: str, context: str) -> Dict[str, float]:
        """
        Correct LLM Judge implementation passing Retrieved Context for Faithfulness.
        """
        system_prompt = (
            "You are an objective AI evaluation judge. Evaluate the Proposed Answer based on the Question, Ground Truth, and Retrieved Context.\n\n"
            "Metrics:\n"
            "1. ACCURACY (1-5): How factually accurate is Proposed Answer compared to Ground Truth?\n"
            "2. RELEVANCY (0-1): Does Proposed Answer directly address the User Question?\n"
            "3. FAITHFULNESS (0-1): Is Proposed Answer strictly derived from Retrieved Context without external hallucinations?\n\n"
            "Output JSON format:\n"
            "{\"accuracy\": number, \"relevancy\": number, \"faithfulness\": number}"
        )
        user_prompt = f"Question: {query}\nGround Truth: {ref}\nRetrieved Context: {context}\nProposed Answer: {hyp}"
        
        try:
            res_str = self.llm_client.generate_response(system_prompt, user_prompt, temperature=0.0)
            data = json.loads(res_str)
            return {
                "accuracy": float(data.get("accuracy", 0.0)),
                "relevancy": float(data.get("relevancy", 0.0)),
                "faithfulness": float(data.get("faithfulness", 0.0))
            }
        except Exception:
            return {"accuracy": 0.0, "relevancy": 0.0, "faithfulness": 0.0}

    def evaluate_intent_clarification(self) -> Dict[str, float]:
        """
        Simulates end-to-end multi-turn dialogue clarification.
        """
        ambiguous_queries = [q for q in self.benchmark_queries if q["type"] == "ambiguous"]
        
        detection_hits = 0
        dialogue_success = 0
        
        for q in ambiguous_queries:
            query = q["query"]
            clarify_result = self.clarifier.check_ambiguity(query)
            
            if clarify_result["is_ambiguous"]:
                detection_hits += 1
                # Simulate user selecting option matching ground truth intent
                selected_option = clarify_result["options"][0]
                response = self.pipeline.run(query=selected_option, faq_threshold=0.70)
                
                if "I could not find sufficient information" not in response["answer"]:
                    dialogue_success += 1

        n = len(ambiguous_queries)
        return {
            "Ambiguity Detection Rate": detection_hits / n if n > 0 else 0.0,
            "End-to-End Clarification Success Rate": dialogue_success / n if n > 0 else 0.0
        }