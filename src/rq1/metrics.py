"""
Evaluation Metrics for RQ1 Experiment
Implements EM (Exact Match), ROUGE-L, and CodeBLEU metrics
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


class MetricsCalculator:
    """Calculate various evaluation metrics for code completion"""

    def __init__(self):
        self.codebleu_available = False
        try:
            from codebleu import calc_codebleu
            self.calc_codebleu = calc_codebleu
            self.codebleu_available = True
        except ImportError:
            print("Warning: codebleu package not available. Install with: pip install codebleu")

    def normalize_code(self, code: str) -> str:
        """
        Normalize code for comparison by removing extra whitespace

        Args:
            code: Code string
        Returns:
            Normalized code
        """
        # Remove leading/trailing whitespace
        code = code.strip()

        # Normalize internal whitespace (but preserve structure)
        lines = code.split('\n')
        normalized_lines = [line.rstrip() for line in lines]

        # Remove consecutive empty lines
        result_lines = []
        prev_empty = False
        for line in normalized_lines:
            is_empty = len(line.strip()) == 0
            if not (is_empty and prev_empty):
                result_lines.append(line)
            prev_empty = is_empty

        return '\n'.join(result_lines)

    def exact_match(self, prediction: str, reference: str, normalize: bool = True) -> bool:
        """
        Calculate exact match score

        Args:
            prediction: Predicted code
            reference: Reference (ground truth) code
            normalize: Whether to normalize before comparison
        Returns:
            True if exact match, False otherwise
        """
        if normalize:
            prediction = self.normalize_code(prediction)
            reference = self.normalize_code(reference)

        return prediction == reference

    def line_exact_match(self, prediction: str, reference: str) -> float:
        """
        Calculate line-by-line exact match accuracy

        Args:
            prediction: Predicted code
            reference: Reference code
        Returns:
            Proportion of lines that match exactly
        """
        pred_lines = prediction.split('\n')
        ref_lines = reference.split('\n')

        max_len = max(len(pred_lines), len(ref_lines))
        if max_len == 0:
            return 1.0

        matches = 0
        for i in range(min(len(pred_lines), len(ref_lines))):
            if pred_lines[i].strip() == ref_lines[i].strip():
                matches += 1

        return matches / max_len

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for ROUGE calculation

        Args:
            text: Input text
        Returns:
            List of tokens
        """
        # Split by whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens

    def _lcs_length(self, X: List[str], Y: List[str]) -> int:
        """
        Compute the length of the longest common subsequence

        Args:
            X, Y: Two sequences
        Returns:
            Length of LCS
        """
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[m][n]

    def rouge_l(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE-L score

        Args:
            prediction: Predicted code
            reference: Reference code
        Returns:
            Dictionary with precision, recall, and f1
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        lcs_len = self._lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def bleu_score(self, prediction: str, reference: str, n: int = 4) -> float:
        """
        Calculate BLEU score (simplified implementation)

        Args:
            prediction: Predicted code
            reference: Reference code
            n: Maximum n-gram size
        Returns:
            BLEU score
        """
        if prediction.strip() == reference.strip():
            return 1.0

        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        max_n = min(n, len(pred_tokens), len(ref_tokens))
        if max_n == 0:
            return 0.0

        precisions = []
        for i in range(1, max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, i)
            ref_ngrams = self._get_ngrams(ref_tokens, i)

            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue

            matches = sum((pred_ngrams & ref_ngrams).values())
            precision = matches / max(sum(pred_ngrams.values()), 1)
            precisions.append(precision)

        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))

        # Simple smoothing to avoid zeroing out when higher-order n-grams mismatch
        epsilon = 1e-9
        precisions = [max(p, epsilon) for p in precisions]
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        return bp * geo_mean

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    def codebleu(
        self,
        prediction: str,
        reference: str,
        lang: str = "python",
        weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    ) -> Dict[str, float]:
        """
        Calculate CodeBLEU score using the codebleu library

        Args:
            prediction: Predicted code
            reference: Reference code
            lang: Programming language
            weights: Weights for (ngram_match, weighted_ngram_match, syntax_match, dataflow_match)
        Returns:
            Dictionary with CodeBLEU scores
        """
        # Handle empty/whitespace-only cases to avoid codebleu errors
        pred_empty = len(prediction.strip()) == 0
        ref_empty = len(reference.strip()) == 0
        if pred_empty or ref_empty:
            val = 1.0 if (pred_empty and ref_empty) else 0.0
            return {
                'codebleu': val,
                'ngram_match_score': val,
                'weighted_ngram_match_score': val,
                'syntax_match_score': val,
                'dataflow_match_score': val
            }

        # Short fragments: skip codebleu library which may error, fallback to smoothed BLEU
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        if min(len(pred_tokens), len(ref_tokens)) < 2:
            bleu = self.bleu_score(prediction, reference)
            return {
                'codebleu': bleu,
                'ngram_match_score': bleu,
                'weighted_ngram_match_score': bleu,
                'syntax_match_score': bleu,
                'dataflow_match_score': bleu
            }

        if not self.codebleu_available:
            print("CodeBLEU not available, using BLEU instead")
            bleu = self.bleu_score(prediction, reference)
            return {
                'codebleu': bleu,
                'ngram_match_score': bleu,
                'weighted_ngram_match_score': 0.0,
                'syntax_match_score': 0.0,
                'dataflow_match_score': 0.0
            }

        try:
            # codebleu expects lists
            predictions = [prediction]
            references = [[reference]]

            result = self.calc_codebleu(
                references,
                predictions,
                lang=lang,
                weights=weights
            )

            return result
        except Exception as e:
            print(f"Error calculating CodeBLEU: {e}")
            bleu = self.bleu_score(prediction, reference)
            # If codebleu fails once, avoid repeated attempts
            self.codebleu_available = False
            return {
                'codebleu': bleu,
                'ngram_match_score': bleu,
                'weighted_ngram_match_score': bleu,
                'syntax_match_score': bleu,
                'dataflow_match_score': bleu
            }

    def evaluate_completion(
        self,
        prediction: str,
        reference: str,
        lang: str = "python"
    ) -> Dict[str, float]:
        """
        Evaluate a completion with all metrics

        Args:
            prediction: Predicted code
            reference: Reference code
            lang: Programming language
        Returns:
            Dictionary with all metric scores
        """
        results = {}

        # Exact Match
        results['exact_match'] = 1.0 if self.exact_match(prediction, reference) else 0.0
        results['line_exact_match'] = self.line_exact_match(prediction, reference)

        # ROUGE-L
        rouge_scores = self.rouge_l(prediction, reference)
        results['rouge_l_precision'] = rouge_scores['precision']
        results['rouge_l_recall'] = rouge_scores['recall']
        results['rouge_l_f1'] = rouge_scores['f1']

        # BLEU
        results['bleu'] = self.bleu_score(prediction, reference)

        # CodeBLEU
        codebleu_scores = self.codebleu(prediction, reference, lang=lang)
        results['codebleu'] = codebleu_scores.get('codebleu', 0.0)
        results['ngram_match'] = codebleu_scores.get('ngram_match_score', 0.0)
        results['weighted_ngram_match'] = codebleu_scores.get('weighted_ngram_match_score', 0.0)
        results['syntax_match'] = codebleu_scores.get('syntax_match_score', 0.0)
        results['dataflow_match'] = codebleu_scores.get('dataflow_match_score', 0.0)

        return results

    def aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across multiple samples

        Args:
            results: List of result dictionaries
        Returns:
            Dictionary with mean and std for each metric
        """
        if not results:
            return {}

        aggregated = {}
        metrics = results[0].keys()

        for metric in metrics:
            values = [r[metric] for r in results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        return aggregated
