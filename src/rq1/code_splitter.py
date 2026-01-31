"""
Code Splitting Strategies for RQ1 Experiment
Implements different strategies to split code at various uncertainty points
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from .uncertainty_calculator import UncertaintyCalculator


class CodeSplitter:
    """Split code at different points based on uncertainty metrics (token-level)"""

    def __init__(self, uncertainty_calculator: UncertaintyCalculator):
        self.uncertainty_calculator = uncertainty_calculator

    def split_at_random_token(self, code: str, seed: int = None) -> Tuple[str, str, Dict]:
        """
        Split code at a random token boundary

        Args:
            code: Complete code string
            seed: Random seed for reproducibility
        Returns:
            (prefix, ground_truth_suffix)
        """
        if seed is not None:
            random.seed(seed)

        token_ids = self.uncertainty_calculator.tokenizer(code, return_tensors="pt").input_ids[0]

        # Valid split points: avoid first and last token to keep both sides non-empty
        valid_split_points = list(range(1, len(token_ids) - 1))

        if not valid_split_points:
            split_idx = len(token_ids) // 2
        else:
            split_idx = random.choice(valid_split_points)

        prefix_ids = token_ids[:split_idx]
        suffix_ids = token_ids[split_idx:]

        prefix = self.uncertainty_calculator.tokenizer.decode(prefix_ids, skip_special_tokens=True)
        suffix = self.uncertainty_calculator.tokenizer.decode(suffix_ids, skip_special_tokens=True)

        info = {
            'split_token_index': split_idx,
            'split_token_text': self.uncertainty_calculator.tokenizer.decode([token_ids[split_idx]]),
            'uncertainty_value': None,
            'metric': 'random',
            'prominence': None
        }

        return prefix, suffix, info

    def split_at_max_uncertainty_token(
        self,
        code: str,
        metric: str = 'entropy',
        min_prominence: float = None,
        prominence_percentile: float = 90.0,
        skip_edges: int = 1
    ) -> Tuple[str, str, Dict]:
        """
        Split code at the token with maximum uncertainty

        Args:
            code: Complete code string
            metric: Uncertainty metric to use ('entropy', 'confidence', 'perplexity')
            min_prominence: Optional fixed prominence threshold (difference to neighbors)
            prominence_percentile: If min_prominence is None, compute this percentile of prominences
            skip_edges: Number of tokens to ignore at each edge when detecting peaks
        Returns:
            (prefix, ground_truth_suffix, uncertainty_info)
        """
        token_uncertainties = self.uncertainty_calculator.compute_uncertainties_for_code(
            code, return_tokens=True
        )

        entropies = token_uncertainties.get('entropies', [])
        confidences = token_uncertainties.get('confidences', [])
        perplexities = token_uncertainties.get('perplexities', [])
        tokens = token_uncertainties.get('tokens', [])

        if metric == 'entropy':
            values = entropies
        elif metric == 'confidence':
            values = [-c for c in confidences]  # invert so higher = more uncertain
        elif metric == 'perplexity':
            values = perplexities
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if not values:
            return code, "", {}

        # Detect prominent peaks (skip edges to avoid trivial first/last token peaks)
        peaks = []
        prominences = []
        for i in range(skip_edges, len(values) - skip_edges):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                prom = values[i] - max(values[i - 1], values[i + 1])
                peaks.append((i, prom, values[i]))
                prominences.append(prom)

        if peaks:
            if min_prominence is None:
                threshold = np.percentile(prominences, prominence_percentile)
            else:
                threshold = min_prominence

            # Filter peaks by prominence
            peaks = [p for p in peaks if p[1] >= threshold]

        # Fallback: if no peaks survive, choose global max
        if peaks:
            # Choose the peak with highest prominence; tie-breaker on value
            peaks.sort(key=lambda x: (x[1], x[2]), reverse=True)
            max_idx, prominence_val, metric_value = peaks[0]
        else:
            max_idx = int(np.argmax(values))
            prominence_val = None
            metric_value = values[max_idx]

        token_ids = self.uncertainty_calculator.tokenizer(code, return_tensors="pt").input_ids[0]

        # Map entropy index to real token position (offset by 1 due to bos handling)
        token_pos = max_idx + 1

        # Split tokens: prefix includes the max-uncertainty token
        prefix_ids = token_ids[: token_pos + 1]
        suffix_ids = token_ids[token_pos + 1 :]
        prefix = self.uncertainty_calculator.tokenizer.decode(prefix_ids, skip_special_tokens=True)
        suffix = self.uncertainty_calculator.tokenizer.decode(suffix_ids, skip_special_tokens=True)

        # Return uncertainty info for analysis
        uncertainty_info = {
            'split_token_index': max_idx,
            'split_token_text': tokens[max_idx] if max_idx < len(tokens) else "",
            'uncertainty_value': metric_value,
            'metric': metric,
            'prominence': prominence_val
        }

        return prefix, suffix, uncertainty_info

    def split_at_all_strategies(
        self,
        code: str,
        seed: int = None
    ) -> Dict[str, Tuple[str, str, Dict]]:
        """
        Split code using all strategies for comparison

        Args:
            code: Complete code string
            seed: Random seed for random strategy
        Returns:
            Dictionary mapping strategy name to (prefix, suffix, info)
        """
        results = {}

        # Random split (token-level)
        prefix, suffix, info = self.split_at_random_token(code, seed=seed)
        results['random'] = (prefix, suffix, info)

        # Entropy-based split
        prefix, suffix, info = self.split_at_max_uncertainty_token(code, metric='entropy')
        results['entropy'] = (prefix, suffix, info)

        # Confidence-based split
        prefix, suffix, info = self.split_at_max_uncertainty_token(code, metric='confidence')
        results['confidence'] = (prefix, suffix, info)

        # Perplexity-based split
        prefix, suffix, info = self.split_at_max_uncertainty_token(code, metric='perplexity')
        results['ppl'] = (prefix, suffix, info)

        # Minimum-uncertainty splits
        prefix, suffix, info = self.split_at_min_uncertainty_token(code, metric='entropy')
        results['entropy_min'] = (prefix, suffix, info)

        prefix, suffix, info = self.split_at_min_uncertainty_token(code, metric='confidence')
        results['confidence_max'] = (prefix, suffix, info)

        prefix, suffix, info = self.split_at_min_uncertainty_token(code, metric='perplexity')
        results['ppl_min'] = (prefix, suffix, info)

        return results

    def get_split_statistics(
        self,
        code: str,
        splits: Dict[str, Tuple[str, str, Dict]]
    ) -> Dict[str, Dict]:
        """
        Get statistics about different splits

        Args:
            code: Original complete code
            splits: Results from split_at_all_strategies
        Returns:
            Statistics for each split strategy
        """
        stats = {}
        total_lines = len(code.split('\n'))

        for strategy, (prefix, suffix, info) in splits.items():
            prefix_lines = len(prefix.split('\n')) if prefix else 0
            suffix_lines = len(suffix.split('\n')) if suffix else 0

            stats[strategy] = {
                'prefix_lines': prefix_lines,
                'suffix_lines': suffix_lines,
                'split_ratio': prefix_lines / total_lines if total_lines > 0 else 0,
                'info': info
            }

        return stats

    def split_at_min_uncertainty_token(
        self,
        code: str,
        metric: str = 'entropy'
    ) -> Tuple[str, str, Dict]:
        """
        Split code at the token with minimum uncertainty (or maximum confidence).

        Args:
            code: Complete code string
            metric: 'entropy' (min), 'perplexity' (min), or 'confidence' (max)
        """
        token_uncertainties = self.uncertainty_calculator.compute_uncertainties_for_code(
            code, return_tokens=True
        )

        entropies = token_uncertainties.get('entropies', [])
        confidences = token_uncertainties.get('confidences', [])
        perplexities = token_uncertainties.get('perplexities', [])
        tokens = token_uncertainties.get('tokens', [])

        if metric == 'entropy':
            values = entropies
            selector = np.argmin
        elif metric == 'perplexity':
            values = perplexities
            selector = np.argmin
        elif metric == 'confidence':
            values = confidences
            selector = np.argmax  # highest confidence = lowest uncertainty
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if not values:
            return code, "", {}

        idx = int(selector(values))
        metric_value = values[idx]

        token_ids = self.uncertainty_calculator.tokenizer(code, return_tensors="pt").input_ids[0]
        token_pos = idx + 1

        prefix_ids = token_ids[: token_pos + 1]
        suffix_ids = token_ids[token_pos + 1 :]
        prefix = self.uncertainty_calculator.tokenizer.decode(prefix_ids, skip_special_tokens=True)
        suffix = self.uncertainty_calculator.tokenizer.decode(suffix_ids, skip_special_tokens=True)

        info = {
            'split_token_index': idx,
            'split_token_text': tokens[idx] if idx < len(tokens) else "",
            'uncertainty_value': metric_value,
            'metric': metric,
            'prominence': None
        }

        return prefix, suffix, info

    def split_at_percentile_uncertainty(
        self,
        code: str,
        metric: str = 'entropy',
        percentile: float = 90.0
    ) -> List[Tuple[str, str, Dict]]:
        """
        Find all tokens above a certain uncertainty percentile and return splits

        Args:
            code: Complete code string
            metric: Uncertainty metric to use
            percentile: Percentile threshold (e.g., 90 for top 10% most uncertain)
        Returns:
            List of (prefix, suffix, info) tuples for high-uncertainty splits
        """
        token_uncertainties = self.uncertainty_calculator.compute_uncertainties_for_code(
            code, return_tokens=True
        )

        if metric == 'entropy':
            uncertainties = token_uncertainties.get('entropies', [])
        elif metric == 'confidence':
            confidences = token_uncertainties.get('confidences', [])
            uncertainties = [1 - c for c in confidences]  # invert so higher = more uncertain
        elif metric == 'perplexity':
            uncertainties = token_uncertainties.get('perplexities', [])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if not uncertainties:
            return []

        threshold = np.percentile(uncertainties, percentile)
        high_uncertainty_indices = [i for i, u in enumerate(uncertainties) if u >= threshold]

        token_ids = self.uncertainty_calculator.tokenizer(code, return_tensors="pt").input_ids[0]
        tokens = token_uncertainties.get('tokens', [])
        splits = []

        for idx in high_uncertainty_indices:
            token_pos = idx + 1
            prefix_ids = token_ids[: token_pos + 1]
            suffix_ids = token_ids[token_pos + 1 :]
            prefix = self.uncertainty_calculator.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            suffix = self.uncertainty_calculator.tokenizer.decode(suffix_ids, skip_special_tokens=True)

            uncertainty_info = {
                'split_token_index': idx,
                'split_token_text': tokens[idx] if idx < len(tokens) else "",
                'uncertainty_value': uncertainties[idx],
                'metric': metric,
                'percentile': percentile
            }

            splits.append((prefix, suffix, uncertainty_info))

        return splits
