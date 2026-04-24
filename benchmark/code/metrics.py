"""
Metrics Calculator

Shared quality metrics evaluation for all engines.
Uses LLM-as-judge, semantic similarity, and ROUGE-L.

Based on the advanced evaluation methodology from GraphRAG Unified Benchmark.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
import logging
import numpy as np
import yaml
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer

import hashlib
import json
import os

LOGGER = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates quality metrics for search results.
    
    Uses a three-pillar evaluation approach:
    1. LLM-as-Judge: Nuanced, criteria-based evaluation
       - Correctness: Factual accuracy vs ground truth
       - Completeness: Coverage of key aspects
       - Relevance: How well it addresses the question
    2. Semantic Similarity: Embedding-based alignment
    3. ROUGE-L: Lexical overlap measurement
    
    Supports two initialization modes:
    - From GraphRAG settings.yaml (LangChain-based)
    - From direct Gemini API key
    """
    
    def __init__(
        self, 
        config_path: Optional[Path] = None, 
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize metrics calculator.
        
        Args:
            config_path: Path to settings.yaml with model config (LangChain mode)
            api_key: Direct API key for Gemini (fallback mode)
            cache_dir: Directory for LLM judge cache
        """
        # Default weights
        self.llm_weight = 0.60
        self.semantic_weight = 0.25
        self.rouge_weight = 0.15
        self.correctness_weight = 0.60
        self.completeness_weight = 0.40
        
        self.llm = None
        self.embeddings = None
        self.llm_available = False
        self._google_client = None
        self._judge_model = "gemini-2.5-flash-lite"
        self._embedding_model = "text-embedding-004"
        
        # Cache settings
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if config_path:
            self._initialize_from_config(config_path)
        elif api_key:
            self._initialize_from_key(api_key)
    
    def _initialize_from_config(self, config_path: Path):
        """Initialize from GraphRAG settings.yaml"""
        try:
            if not config_path.exists():
                print(f"⚠️  MetricsCalculator: Config not found at {config_path}")
                return
            
            with open(config_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Load evaluation weights if present
            eval_cfg = settings.get('evaluation', {})
            self.llm_weight = eval_cfg.get('llm_weight', self.llm_weight)
            self.semantic_weight = eval_cfg.get('semantic_weight', self.semantic_weight)
            self.rouge_weight = eval_cfg.get('rouge_weight', self.rouge_weight)
            self.correctness_weight = eval_cfg.get('correctness_weight', self.correctness_weight)
            self.completeness_weight = eval_cfg.get('completeness_weight', self.completeness_weight)
            
            models_cfg = settings.get('models', {})
            chat_config = models_cfg.get('default_chat_model', {})
            api_base = chat_config.get('api_base')
            chat_model = chat_config.get('model')
            api_key = chat_config.get('api_key')
            
            # Allow judge model override
            self._judge_model = eval_cfg.get('judge_model', chat_model or self._judge_model)
            
            embedding_config = models_cfg.get('default_embedding_model', {})
            embedding_model = embedding_config.get('model')
            self._embedding_model = eval_cfg.get('embedding_model', embedding_model or self._embedding_model)
            
            if not all([api_base, chat_model, embedding_model, api_key]):
                print("⚠️  MetricsCalculator: Incomplete model config in settings.yaml")
                return
            
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            self.llm = ChatOpenAI(
                base_url=api_base,
                api_key=api_key,
                model=chat_model,
                temperature=0.0
            )
            self.embeddings = OpenAIEmbeddings(
                base_url=api_base,
                api_key=api_key,
                model=embedding_model
            )
            self.llm_available = True
            print(f"✅ MetricsCalculator initialized with {self._judge_model} judge")
            
        except Exception as e:
            print(f"⚠️  MetricsCalculator: Could not initialize from config: {e}")
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("Metrics config initialization failed")
    
    def _initialize_from_key(self, api_key: str):
        """Initialize with direct Gemini API key using google-genai client"""
        try:
            from google import genai
            self._google_client = genai.Client(api_key=api_key)
            self._api_key = api_key
            self.llm_available = True
            print("✅ MetricsCalculator initialized with google-genai")
            
        except Exception as e:
            print(f"⚠️  MetricsCalculator: Could not initialize google-genai: {e}")
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("Metrics google-genai initialization failed")
    
    # =========================================================================
    # PYDANTIC MODELS FOR STRUCTURED OUTPUT
    # =========================================================================
    
    class AnswerEvaluation(BaseModel):
        """Structured output for LLM-as-Judge evaluation"""
        correctness: float = Field(
            description="Score from 0.0 to 1.0 assessing if the answer is factually "
                       "correct and consistent with the ground truth. 1.0 means fully correct."
        )
        completeness: float = Field(
            description="Score from 0.0 to 1.0 assessing if the answer covers all key "
                       "aspects mentioned in the ground truth. 1.0 means all aspects covered."
        )
        relevance: float = Field(
            description="Score from 0.0 to 1.0 assessing if the answer directly and "
                       "appropriately addresses the user's question. 1.0 means perfectly relevant."
        )
        reasoning: str = Field(
            description="A detailed step-by-step reasoning for the scores provided, "
                       "explaining the assessment of correctness, completeness, and relevance."
        )
    
    class ContextRelevance(BaseModel):
        """Structured output for context relevance evaluation"""
        score: float = Field(
            description="Relevance score from 0.0 (completely irrelevant) to 1.0 "
                       "(perfectly relevant and sufficient)."
        )
        reasoning: str = Field(
            description="Brief reasoning explaining the relevance assessment."
        )
    
    # =========================================================================
    # MAIN PUBLIC METHOD
    # =========================================================================
    
    async def calculate_all_metrics(
        self,
        answer: str,
        ground_truth: str,
        question: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate all quality metrics for a search result.
        
        Uses a holistic three-pillar approach:
        1. LLM-as-Judge for nuanced criteria-based evaluation
        2. Embedding similarity for semantic alignment
        3. ROUGE-L for lexical overlap
        
        Args:
            answer: Generated answer from the search engine
            ground_truth: Expected/reference answer
            question: Original question asked
            context: Retrieved context (optional, for context relevance)
            
        Returns:
            Dictionary containing:
            - answer_accuracy: Combined weighted score (0-1)
            - llm_judged_correctness: Factual accuracy score (0-1)
            - llm_judged_completeness: Coverage score (0-1)
            - llm_judged_relevance: Relevance to question (0-1)
            - llm_judged_reasoning: Detailed reasoning string
            - semantic_similarity: Embedding cosine similarity (0-1)
            - rougeL_f1: ROUGE-L F1 score (0-1)
            - context_relevance: Context quality score (0-1, if context provided)
        """
        if not self.llm_available:
            return {}
        
        metrics = {}
        
        try:
            # Main accuracy calculation using three-pillar approach
            accuracy_results = await self._calculate_accuracy(answer, ground_truth, question)
            metrics.update(accuracy_results)
            
            # Context relevance (if context provided)
            if context:
                metrics['context_relevance'] = await self._calculate_context_relevance(
                    question, context
                )
        except Exception as e:
            print(f"⚠️  Error calculating quality metrics: {e}")
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("Quality metrics calculation failed")
        
        return metrics
    
    # =========================================================================
    # CACHING UTILITIES
    # =========================================================================

    def _get_cache_key(self, question: str, prediction: str, ground_truth: str) -> str:
        """Generate a unique key for an evaluation triple."""
        data = f"{question}|{prediction}|{ground_truth}|{self._judge_model}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation from disk cache."""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, key: str, result: Dict[str, Any]):
        """Save evaluation result to disk cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass

    # =========================================================================
    # ACCURACY CALCULATION (THREE-PILLAR APPROACH)
    # =========================================================================
    
    async def _calculate_accuracy(
        self,
        prediction: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Calculate holistic answer accuracy using three pillars.
        Checks cache first before calling LLM.
        """
        cache_key = self._get_cache_key(question, prediction, ground_truth)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Route to appropriate backend
        if hasattr(self, 'llm') and self.llm is not None:
            result = await self._calculate_accuracy_langchain(prediction, ground_truth, question)
        else:
            result = await self._calculate_accuracy_gemini(prediction, ground_truth, question)
        
        self._save_to_cache(cache_key, result)
        return result
    
    async def _calculate_accuracy_langchain(
        self,
        prediction: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, Any]:
        """Calculate accuracy using LangChain models (OpenAI-compatible API)"""
        
        # Build evaluation prompt
        eval_prompt = self._build_eval_prompt(question, ground_truth, prediction)
        
        # LLM-as-Judge evaluation with structured output
        evaluator = self.llm.with_structured_output(self.AnswerEvaluation)
        llm_result = await evaluator.ainvoke(eval_prompt)
        
        # Semantic similarity via embeddings
        embeddings = await self.embeddings.aembed_documents([prediction, ground_truth])
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        
        # ROUGE-L lexical overlap
        rouge_score = self._calculate_rouge(prediction, ground_truth)
        
        # Combine scores using weighted formula
        llm_score = (self.correctness_weight * llm_result.correctness + 
                     self.completeness_weight * llm_result.completeness)
        combined = (self.llm_weight * llm_score + 
                    self.semantic_weight * similarity + 
                    self.rouge_weight * rouge_score)
        
        return {
            "answer_accuracy": combined,
            "llm_judged_correctness": llm_result.correctness,
            "llm_judged_completeness": llm_result.completeness,
            "llm_judged_relevance": llm_result.relevance,
            "llm_judged_reasoning": llm_result.reasoning,
            "semantic_similarity": similarity,
            "rougeL_f1": rouge_score,
        }
    
    async def _calculate_accuracy_gemini(
        self,
        prediction: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, Any]:
        """Calculate accuracy using direct Gemini API (fallback mode)"""
        import json

        if self._google_client is None:
            raise RuntimeError("google-genai client is not initialized")
        
        # Build evaluation prompt with JSON output instruction
        eval_prompt = self._build_eval_prompt(question, ground_truth, prediction)
        eval_prompt += """

Return your evaluation as JSON with this exact format:
{
    "correctness": <float 0-1>,
    "completeness": <float 0-1>,
    "relevance": <float 0-1>,
    "reasoning": "<string>"
}"""
        
        # LLM-as-Judge evaluation with retry logic for fault tolerance
        max_retries = 3
        llm_result = None
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self._google_client.models.generate_content,
                    model=self._judge_model,
                    contents=eval_prompt,
                )
                
                # Parse JSON from response
                text = getattr(response, "text", "") or ""
                start = text.find('{')
                end = text.rfind('}') + 1
                
                if start == -1 or end == 0:
                    raise ValueError("No JSON object found in response")
                
                json_str = text[start:end]
                llm_result = json.loads(json_str)
                
                # Ensure numeric values are floats (LLM might return strings like "0.7")
                llm_result['correctness'] = float(llm_result.get('correctness', 0.0))
                llm_result['completeness'] = float(llm_result.get('completeness', 0.0))
                llm_result['relevance'] = float(llm_result.get('relevance', 0.0))
                
                # Success - exit retry loop
                break
                
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  LLM judge response parse failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                else:
                    print(f"⚠️  LLM judge response parse failed after {max_retries} attempts: {e}")
                    llm_result = {
                        "correctness": 0.0, 
                        "completeness": 0.0, 
                        "relevance": 0.0, 
                        "reasoning": f"Failed to parse LLM response after {max_retries} attempts"
                    }
        
        # Semantic similarity via google-genai embeddings
        pred_emb_response = await asyncio.to_thread(
            self._google_client.models.embed_content,
            model=self._embedding_model,
            contents=prediction,
        )
        gt_emb_response = await asyncio.to_thread(
            self._google_client.models.embed_content,
            model=self._embedding_model,
            contents=ground_truth,
        )

        pred_emb = self._extract_google_embedding(pred_emb_response)
        gt_emb = self._extract_google_embedding(gt_emb_response)
        
        similarity = self._cosine_similarity(pred_emb, gt_emb)
        
        # ROUGE-L lexical overlap
        rouge_score = self._calculate_rouge(prediction, ground_truth)
        
        # Combine scores using weighted formula
        llm_score = (self.correctness_weight * llm_result['correctness'] + 
                     self.completeness_weight * llm_result['completeness'])
        combined = (self.llm_weight * llm_score + 
                    self.semantic_weight * similarity + 
                    self.rouge_weight * rouge_score)
        
        return {
            "answer_accuracy": combined,
            "llm_judged_correctness": llm_result['correctness'],
            "llm_judged_completeness": llm_result['completeness'],
            "llm_judged_relevance": llm_result['relevance'],
            "llm_judged_reasoning": llm_result['reasoning'],
            "semantic_similarity": similarity,
            "rougeL_f1": rouge_score,
        }
    
    # =========================================================================
    # EVALUATION PROMPT BUILDER
    # =========================================================================
    
    def _build_eval_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """
        Build comprehensive evaluation prompt for LLM-as-judge.
        
        This prompt is designed to elicit nuanced, step-by-step evaluation
        of the generated answer against the ground truth reference.
        """
        return f"""You are an impartial and scientific evaluator. Your task is to assess a generated answer against a ground truth reference, based on a specific question. Evaluate the answer based on the criteria of Correctness, Completeness, and Relevance.

**Evaluation Task:**

1.  **Analyze the Question:** Understand what the user is asking for.
    - Question: "{question}"

2.  **Analyze the Ground Truth:** This is the reference for what a correct and complete answer should contain.
    - Ground Truth: "{ground_truth}"

3.  **Analyze the Generated Answer:** This is the answer you must evaluate.
    - Generated Answer: "{prediction}"

4.  **Perform a Step-by-Step Evaluation:**

    - **Correctness (0.0-1.0):** Is the information in the Generated Answer factually accurate and consistent with the Ground Truth?
      - Does it contradict the ground truth?
      - Does it introduce plausible but unsupported information (hallucinations)?
      - Note: A correct generalization or abstraction of the ground truth is still considered correct.

    - **Completeness (0.0-1.0):** Does the Generated Answer cover all the key information and essential points present in the Ground Truth?
      - Identify the core concepts in the ground truth.
      - Check if these concepts are present in the generated answer, even if phrased differently.

    - **Relevance (0.0-1.0):** Does the Generated Answer directly address the user's Question?
      - Is the answer on-topic?
      - Does it include superfluous information that detracts from the main point?
      - Note: Providing relevant, illustrative examples that enhance the explanation should NOT be penalized and can contribute to a high relevance score.

5.  **Provide Scores and Reasoning:** Based on your analysis, provide a score from 0.0 (terrible) to 1.0 (perfect) for each criterion and write a detailed reasoning for your scores."""
    
    # =========================================================================
    # CONTEXT RELEVANCE EVALUATION
    # =========================================================================
    
    async def _calculate_context_relevance(self, question: str, context: str) -> float:
        """
        Calculate how relevant the retrieved context is for answering the question.
        
        Uses structured output when available for more reliable scoring.
        """
        relevance_prompt = f"""You are evaluating the relevance of retrieved context for answering a question.

Question: {question}

Retrieved Context:
{context}

Evaluation Criteria:
1. Does the context contain information directly relevant to answering the question?
2. Is the information sufficient to construct a complete answer?
3. How much of the context is pertinent vs. irrelevant?

Provide a relevance score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and sufficient).
Also provide brief reasoning for your score."""
        
        if hasattr(self, 'llm') and self.llm is not None:
            # Use structured output with LangChain
            try:
                evaluator = self.llm.with_structured_output(self.ContextRelevance)
                result = await evaluator.ainvoke(relevance_prompt)
                return result.score
            except Exception:
                if LOGGER.isEnabledFor(logging.INFO):
                    LOGGER.exception("Structured context relevance evaluation failed")
                # Fallback to simple invocation
                response = await self.llm.ainvoke(relevance_prompt)
                try:
                    return float(response.content.strip())
                except (ValueError, AttributeError):
                    return 0.5
        else:
            # google-genai fallback
            if self._google_client is None:
                return 0.5
            response = await asyncio.to_thread(
                self._google_client.models.generate_content,
                model=self._judge_model,
                contents=relevance_prompt + "\n\nReturn only a number between 0 and 1.",
            )
            try:
                return float((getattr(response, "text", "") or "").strip())
            except (ValueError, AttributeError):
                return 0.5

    def _extract_google_embedding(self, response: Any) -> list[float]:
        """Extract embedding values across google-genai response shapes."""
        if response is None:
            return []

        if hasattr(response, "embeddings") and response.embeddings:
            first = response.embeddings[0]
            if hasattr(first, "values"):
                return list(first.values)
            if isinstance(first, dict) and "values" in first:
                return list(first["values"])

        if hasattr(response, "embedding") and response.embedding:
            emb = response.embedding
            if hasattr(emb, "values"):
                return list(emb.values)
            if isinstance(emb, dict) and "values" in emb:
                return list(emb["values"])

        if isinstance(response, dict):
            embeddings = response.get("embeddings")
            if embeddings and isinstance(embeddings, list):
                first = embeddings[0]
                if isinstance(first, dict) and "values" in first:
                    return list(first["values"])
            embedding = response.get("embedding")
            if isinstance(embedding, dict) and "values" in embedding:
                return list(embedding["values"])

        return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns value in [0, 1] range for consistency with other metrics.
        """
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity: dot product / (norm1 * norm2)
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Clamp to [0, 1] - text embeddings rarely produce negative similarity
        # Using max(0, sim) instead of (sim + 1) / 2 prevents score inflation
        return max(0.0, similarity)
    
    def _calculate_rouge(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate ROUGE-L F1 score for lexical overlap.
        
        ROUGE-L uses longest common subsequence to measure overlap,
        which is more flexible than exact n-gram matching.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        
        return scores['rougeL'].fmeasure
