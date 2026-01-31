"""
Uncertainty Calculator Module for RQ1 Experiment
Computes different types of uncertainties (Entropy, Confidence, PPL) for code tokens
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class UncertaintyCalculator:
    """Calculate various uncertainty metrics for model predictions"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate Shannon entropy of token distribution
        Higher entropy = higher uncertainty
        Note: This implementation is consistent with Generator.calculate_entropy()

        Args:
            logits: Raw logits from model [vocab_size] or [batch_size, vocab_size]
        Returns:
            Entropy value(s)
        """
        # 与Generator类中的方法保持一致
        next_token_probs_exp = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(next_token_probs_exp * log_probs, dim=-1)
        return entropy

    def calculate_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence as the maximum probability
        Lower confidence = higher uncertainty

        Args:
            logits: Raw logits from model [vocab_size] or [batch_size, vocab_size]
        Returns:
            Confidence value(s) (max probability)
        """
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        return confidence

    def calculate_perplexity(self, logits: torch.Tensor, target_token: torch.Tensor) -> torch.Tensor:
        """
        Calculate perplexity for the target token
        Higher perplexity = higher uncertainty

        Args:
            logits: Raw logits from model [vocab_size] or [batch_size, vocab_size]
            target_token: Ground truth token id(s)
        Returns:
            Perplexity value(s)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        if logits.dim() == 1:
            # Single token case
            target_log_prob = log_probs[target_token]
        else:
            # Batch case
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            target_log_prob = log_probs[batch_indices, target_token]

        perplexity = torch.exp(-target_log_prob)
        return perplexity

    def compute_uncertainties_for_code(
        self,
        code: str,
        return_tokens: bool = True
    ) -> Dict[str, List]:
        """
        Compute all uncertainty metrics for each token in the code

        Args:
            code: Complete code string
            return_tokens: Whether to return decoded tokens
        Returns:
            Dictionary with lists of uncertainties and optionally tokens
        """
        # Tokenize the code
        token_ids = self.tokenizer(code, return_tensors="pt").input_ids[0].to(self.device)

        entropies = []
        confidences = []
        perplexities = []
        tokens = []

        with torch.no_grad():
            # Process each position
            for i in range(1, len(token_ids)):
                # Get context (everything before current token)
                context_ids = token_ids[:i].unsqueeze(0)

                # Get model prediction
                outputs = self.model(context_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits

                # Target is the actual next token
                target_token = token_ids[i]

                # Calculate all uncertainties
                entropy = self.calculate_entropy(logits)
                confidence = self.calculate_confidence(logits)
                perplexity = self.calculate_perplexity(logits, target_token)

                entropies.append(entropy.item())
                confidences.append(confidence.item())
                perplexities.append(perplexity.item())

                if return_tokens:
                    token_text = self.tokenizer.decode([target_token.item()])
                    tokens.append(token_text)

        result = {
            'entropies': entropies,
            'confidences': confidences,
            'perplexities': perplexities,
        }

        if return_tokens:
            result['tokens'] = tokens
            result['token_ids'] = token_ids[1:].cpu().tolist()

        return result

    def find_line_boundaries(self, code: str, token_positions: List[int]) -> List[int]:
        """
        Find which tokens correspond to line boundaries (newlines)

        Args:
            code: Complete code string
            token_positions: List of token indices
        Returns:
            List of token indices that are at line boundaries
        """
        tokens = self.tokenizer(code, return_tensors="pt").input_ids[0]
        line_boundary_indices = []

        for i, token_id in enumerate(tokens[1:], start=1):  # Skip first token
            token_text = self.tokenizer.decode([token_id.item()])
            if '\n' in token_text:
                line_boundary_indices.append(i - 1)  # Adjust for 0-based indexing

        return line_boundary_indices

    def aggregate_uncertainties_by_line(
        self,
        code: str
    ) -> Dict[str, List]:
        """
        Compute average uncertainties for each line of code

        Args:
            code: Complete code string
        Returns:
            Dictionary with line-level aggregated uncertainties
        """
        # Get token-level uncertainties
        uncertainties = self.compute_uncertainties_for_code(code, return_tokens=True)

        # Split code into lines
        lines = code.split('\n')

        # Tokenize and track which tokens belong to which line
        current_pos = 0
        line_uncertainties = {
            'line_numbers': [],
            'line_contents': [],
            'avg_entropies': [],
            'avg_confidences': [],
            'avg_perplexities': [],
            'max_entropies': [],
            'min_confidences': [],
            'max_perplexities': []
        }

        for line_idx, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue

            # Find tokens for this line
            line_tokens = self.tokenizer(line, return_tensors="pt").input_ids[0]
            num_tokens = len(line_tokens) - 1  # Exclude special tokens

            if num_tokens > 0 and current_pos + num_tokens <= len(uncertainties['entropies']):
                # Extract uncertainties for this line
                line_entropies = uncertainties['entropies'][current_pos:current_pos + num_tokens]
                line_confidences = uncertainties['confidences'][current_pos:current_pos + num_tokens]
                line_perplexities = uncertainties['perplexities'][current_pos:current_pos + num_tokens]

                # Aggregate
                line_uncertainties['line_numbers'].append(line_idx)
                line_uncertainties['line_contents'].append(line)
                line_uncertainties['avg_entropies'].append(np.mean(line_entropies) if line_entropies else 0)
                line_uncertainties['avg_confidences'].append(np.mean(line_confidences) if line_confidences else 0)
                line_uncertainties['avg_perplexities'].append(np.mean(line_perplexities) if line_perplexities else 0)
                line_uncertainties['max_entropies'].append(max(line_entropies) if line_entropies else 0)
                line_uncertainties['min_confidences'].append(min(line_confidences) if line_confidences else 1.0)
                line_uncertainties['max_perplexities'].append(max(line_perplexities) if line_perplexities else 0)

                current_pos += num_tokens

        return line_uncertainties
