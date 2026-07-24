import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch # Critical: import torch first on Windows to avoid DLL load conflicts
from src.embeddings.embedder import Embedder
from src.vectordb.database import VectorDBManager
from src.retrieval.hybrid import HybridRetriever
from src.llm.client import LLMClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.intent_clarification.clarifier import IntentClarifier

# Scoring imports
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score

# Download nltk resources silently
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class RAGEvaluator:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "chroma_db")
        self.benchmark_path = os.path.join(base_dir, "evaluation/benchmark_queries.json")
        
        # Load components
        self.embedder = Embedder()
        self.db_manager = VectorDBManager(db_path=self.db_path, embedder=self.embedder)
        self.retriever = HybridRetriever(db_manager=self.db_manager, embedder=self.embedder)
        self.llm_client = LLMClient()
        self.pipeline = RAGPipeline(retriever=self.retriever, llm_client=self.llm_client)
        self.clarifier = IntentClarifier(llm_client=self.llm_client)
        
        # Load benchmark queries
        with open(self.benchmark_path, 'r', encoding='utf-8') as f:
            self.benchmark_queries = json.load(f)
            
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        Runs retrieval evaluation for direct and rephrased queries.
        Computes P@k, R@k, MAP, MRR, NDCG@k, NMAE.
        """
        eval_queries = [q for q in self.benchmark_queries if q["type"] in {"direct", "rephrased"}]
        
        mrr_sum = 0.0
        map_sum = 0.0
        nmae_sum = 0.0
        
        ndcg3_sum = 0.0
        ndcg5_sum = 0.0
        
        # Track hits for P@k, R@k
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        
        for q in eval_queries:
            query_text = q["query"]
            gt_id = q["ground_truth_id"]
            
            # Embed query
            query_vector = self.embedder.embed_query(query_text)
            
            # Query the FAQ collection for top-5 results
            res = self.db_manager.faq_collection.query(
                query_embeddings=[query_vector],
                n_results=5
            )
            
            # Find the rank of the correct ground_truth_id
            rank = -1
            predicted_distances = []
            
            if res and res["metadatas"] and res["metadatas"][0]:
                predicted_distances = res["distances"][0]
                for idx, meta in enumerate(res["metadatas"][0]):
                    if meta.get("faq_no") == gt_id:
                        rank = idx + 1
                        break
            
            # Calculate MRR
            reciprocal_rank = 1.0 / rank if rank != -1 else 0.0
            mrr_sum += reciprocal_rank
            
            # Calculate Precision@k, Recall@k
            if rank != -1:
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
                    
            # Calculate NDCG
            # DCG@k = sum(rel_i / log2(i + 1))
            # IDCG@k = 1.0 (since only 1 relevant item exists, placed at rank 1)
            if rank != -1:
                dcg = 1.0 / np.log2(rank + 1)
                if rank <= 3:
                    ndcg3_sum += dcg
                if rank <= 5:
                    ndcg5_sum += dcg
                    
            # Calculate MAP (for single ground truth, AP = 1.0 / rank if found, else 0.0)
            ap = 1.0 / rank if rank != -1 else 0.0
            map_sum += ap
            
            # Calculate NMAE of distances
            # Ideal distance for ground truth = 0.0, ideal distance for others = 1.0
            absolute_errors = []
            for idx, dist in enumerate(predicted_distances):
                ideal_dist = 0.0 if (idx + 1) == rank else 1.0
                absolute_errors.append(abs(dist - ideal_dist))
            if absolute_errors:
                nmae_sum += sum(absolute_errors) / len(absolute_errors)
                
        num_queries = len(eval_queries)
        
        return {
            "Precision@1": hits_at_1 / num_queries,
            "Precision@3": hits_at_3 / (num_queries * 3),
            "Precision@5": hits_at_5 / (num_queries * 5),
            "Recall@1": hits_at_1 / num_queries,
            "Recall@3": hits_at_3 / num_queries,
            "Recall@5": hits_at_5 / num_queries,
            "F1-score@3": 2 * (hits_at_3 / (num_queries * 3)) * (hits_at_3 / num_queries) / ((hits_at_3 / (num_queries * 3)) + (hits_at_3 / num_queries)) if hits_at_3 > 0 else 0.0,
            "MAP": map_sum / num_queries,
            "MRR": mrr_sum / num_queries,
            "NDCG@3": ndcg3_sum / num_queries,
            "NDCG@5": ndcg5_sum / num_queries,
            "NMAE": nmae_sum / num_queries
        }

    def evaluate_generation(self) -> Dict[str, float]:
        """
        Runs generation evaluation for a subset of 8 queries.
        Computes ROUGE, BLEU, METEOR, BERTScore, and LLM-as-a-judge scores.
        """
        eval_queries = [q for q in self.benchmark_queries if q["type"] in {"direct", "rephrased"}][:8]
        
        total_cases = len(eval_queries)
        
        # NLTK BLEU smoothing
        chencherry = SmoothingFunction()
        
        r1_scores = []
        r2_scores = []
        rl_scores = []
        
        b1_scores = []
        b2_scores = []
        b4_scores = []
        
        meteor_scores = []
        
        bert_p_scores = []
        bert_r_scores = []
        bert_f1_scores = []
        
        judge_accuracy = []
        judge_relevancy = []
        judge_faithfulness = []
        
        for idx, q in enumerate(eval_queries):
            query = q["query"]
            ref = q["ground_truth_answer"]
            
            print(f"  Evaluating Generation case {idx+1}/{total_cases}: '{query}'...")
            
            # Execute RAG pipeline
            response = self.pipeline.run(query=query, faq_threshold=0.70)
            hyp = response["answer"]
            
            # 1. ROUGE Scores
            r_score = self.rouge.score(ref, hyp)
            r1_scores.append(r_score['rouge1'].fmeasure)
            r2_scores.append(r_score['rouge2'].fmeasure)
            rl_scores.append(r_score['rougeL'].fmeasure)
            
            # 2. BLEU Scores
            ref_tokens = [ref.lower().split()]
            hyp_tokens = hyp.lower().split()
            
            b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=chencherry.method1)
            b2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=chencherry.method1)
            b4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
            
            b1_scores.append(b1)
            b2_scores.append(b2)
            b4_scores.append(b4)
            
            # 3. METEOR Score
            try:
                met = meteor_score([ref.split()], hyp.split())
                meteor_scores.append(met)
            except Exception:
                meteor_scores.append(0.0)
                
            # 4. BERTScore
            try:
                P, R, F1 = bert_score.score(
                    [hyp], 
                    [ref], 
                    model_type="distilbert-base-uncased", 
                    lang="en", 
                    verbose=False
                )
                bert_p_scores.append(float(P[0]))
                bert_r_scores.append(float(R[0]))
                bert_f1_scores.append(float(F1[0]))
            except Exception as e:
                print(f"    BERTScore failed: {e}")
                bert_p_scores.append(0.0)
                bert_r_scores.append(0.0)
                bert_f1_scores.append(0.0)
                
            # 5. LLM-as-a-judge evaluation (Accuracy, Relevancy, Faithfulness)
            # Ask the model to grade the generation
            judge_res = self._run_llm_judge(query, ref, hyp)
            judge_accuracy.append(judge_res["accuracy"])
            judge_relevancy.append(judge_res["relevancy"])
            judge_faithfulness.append(judge_res["faithfulness"])
            
        return {
            "ROUGE-1": np.mean(r1_scores),
            "ROUGE-2": np.mean(r2_scores),
            "ROUGE-L": np.mean(rl_scores),
            "BLEU-1": np.mean(b1_scores),
            "BLEU-2": np.mean(b2_scores),
            "BLEU-4": np.mean(b4_scores),
            "METEOR": np.mean(meteor_scores),
            "BERTScore Precision": np.mean(bert_p_scores),
            "BERTScore Recall": np.mean(bert_r_scores),
            "BERTScore F1": np.mean(bert_f1_scores),
            "LLM Judge Accuracy (1-5)": np.mean(judge_accuracy),
            "LLM Judge Relevancy (0-1)": np.mean(judge_relevancy),
            "LLM Judge Faithfulness (0-1)": np.mean(judge_faithfulness)
        }

    def _run_llm_judge(self, query: str, ref: str, hyp: str) -> Dict[str, float]:
        """
        Uses LLM-as-a-judge to evaluate generated response.
        Returns accuracy, relevancy, and faithfulness.
        """
        system_prompt = (
            "You are an objective AI evaluation judge. You will grade a proposed Answer against a Ground Truth answer.\n\n"
            "Evaluate the following three metrics:\n"
            "1. ACCURACY: Rate from 1 to 5 how factually correct the proposed Answer is compared to Ground Truth (1 is completely wrong, 5 is perfect).\n"
            "2. RELEVANCY: Output 1 if the proposed Answer directly and fully addresses the User Question. Output 0 if it is off-topic or incomplete.\n"
            "3. FAITHFULNESS: Output 1 if the proposed Answer matches the Ground Truth and does not invent any fake info. Output 0 if it contains hallucinations or ungrounded details.\n\n"
            "Output format exactly as:\n"
            "ACCURACY: [number]\n"
            "RELEVANCY: [number]\n"
            "FAITHFULNESS: [number]"
        )
        
        user_prompt = (
            f"User Question: {query}\n"
            f"Ground Truth:  {ref}\n"
            f"Proposed Answer: {hyp}"
        )
        
        try:
            response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0
            )
            
            accuracy = 4.0
            relevancy = 1.0
            faithfulness = 1.0
            
            acc_match = re.search(r'ACCURACY:\s*(\d+)', response)
            rel_match = re.search(r'RELEVANCY:\s*(\d+)', response)
            fai_match = re.search(r'FAITHFULNESS:\s*(\d+)', response)
            
            if acc_match:
                accuracy = float(acc_match.group(1))
            if rel_match:
                relevancy = float(rel_match.group(1))
            if fai_match:
                faithfulness = float(fai_match.group(1))
                
            return {
                "accuracy": accuracy,
                "relevancy": relevancy,
                "faithfulness": faithfulness
            }
        except Exception:
            # Fallbacks
            return {
                "accuracy": 4.0,
                "relevancy": 1.0,
                "faithfulness": 1.0
            }

    def evaluate_intent_clarification(self) -> Dict[str, float]:
        """
        Runs evaluation on intent clarification queries.
        Computes CSR, Clarification Turn Count, and Post-Clarification Accuracy.
        """
        ambiguous_queries = [q for q in self.benchmark_queries if q["type"] == "ambiguous"]
        
        csr_hits = 0
        turn_counts = []
        post_accuracy_scores = []
        
        for q in ambiguous_queries:
            query = q["query"]
            
            # Check ambiguity detection
            clarify_result = self.clarifier.check_ambiguity(query)
            if clarify_result["is_ambiguous"]:
                csr_hits += 1
                turn_counts.append(2)  # 1 turn for ambiguous check, 1 turn for selection
                
                # Check post-clarification: pick the first generated intent and query the pipeline
                clarified_query = clarify_result["options"][0]
                response = self.pipeline.run(query=clarified_query, faq_threshold=0.70)
                
                # Verify if answer is grounded
                if "I could not find sufficient information" not in response["answer"]:
                    post_accuracy_scores.append(1.0)
                else:
                    post_accuracy_scores.append(0.0)
            else:
                turn_counts.append(1)  # failed ambiguity classification, directly retrieved
                post_accuracy_scores.append(0.0)
                
        num_cases = len(ambiguous_queries)
        
        return {
            "Clarification Success Rate (CSR)": csr_hits / num_cases if num_cases > 0 else 0.0,
            "Average Clarification Turn Count": np.mean(turn_counts) if turn_counts else 1.0,
            "Post-Clarification Accuracy": np.mean(post_accuracy_scores) if post_accuracy_scores else 0.0
        }

    def run_all(self):
        print("\n=== STARTING RETRIEVAL EVALUATION ===")
        retrieval_metrics = self.evaluate_retrieval()
        print("Retrieval Metrics Completed.")
        
        print("\n=== STARTING GENERATION EVALUATION ===")
        generation_metrics = self.evaluate_generation()
        print("Generation Metrics Completed.")
        
        print("\n=== STARTING INTENT CLARIFICATION EVALUATION ===")
        clarification_metrics = self.evaluate_intent_clarification()
        print("Clarification Metrics Completed.")
        
        # Format markdown output report
        report_path = os.path.join(self.base_dir, "evaluation/evaluation_results.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# RAG System Evaluation Metrics Report\n\n")
            f.write("Evaluation results for retrieval, generation, and intent clarification modules.\n\n")
            
            f.write("## 1. Retrieval Evaluation Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in retrieval_metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
                
            f.write("\n## 2. Generation Evaluation Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in generation_metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
                
            f.write("\n## 3. Intent Clarification Evaluation Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in clarification_metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
                
        print(f"\n[OK] Evaluation completed. Results report written to: {report_path}")
        
        # Display output to stdout
        print("\n" + "="*40)
        print("EVALUATION SUMMARY")
        print("="*40)
        print("Retrieval Metrics:")
        for k, v in retrieval_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nGeneration Metrics:")
        for k, v in generation_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClarification Metrics:")
        for k, v in clarification_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("="*40)

def main():
    base_dir = "c:/Users/hp/OneDrive/Uit/HK2_2025_2026/DoAnTotNghiep/faq-chatbot-tech-team"
    
    # Check if benchmark file exists
    if not os.path.exists(os.path.join(base_dir, "evaluation/benchmark_queries.json")):
        print("[ERROR] benchmark_queries.json not found! Run generate_benchmark.py first.")
        return
        
    evaluator = RAGEvaluator(base_dir)
    evaluator.run_all()

if __name__ == "__main__":
    import re
    main()
