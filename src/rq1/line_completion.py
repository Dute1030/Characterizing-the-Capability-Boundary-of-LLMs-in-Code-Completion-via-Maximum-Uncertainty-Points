"""
Line-level Code Completion for RQ1 Experiment
Performs code completion and tracks performance
"""

import torch
import re
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class LineCompletion:
    """Perform line-level code completion"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def complete_line(
        self,
        prefix: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_at_newline: bool = True
    ) -> str:
        """
        Complete code from the given prefix (generate one line)

        Args:
            prefix: Code prefix (context before cursor)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_at_newline: Stop generation at newline
        Returns:
            Generated completion string
        """
        # Tokenize prefix
        input_ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            if stop_at_newline:
                # Greedy decode token by token until newline
                completion = self._generate_until_newline(
                    input_ids,
                    max_new_tokens
                )
            else:
                # Greedy generate using HF generate
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                completion = self.tokenizer.decode(
                    output_ids[0][len(input_ids[0]):],
                    skip_special_tokens=True
                )

        return completion

    def _generate_until_newline(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int
    ) -> str:
        """Greedy-generate tokens until a newline is encountered"""
        generated_tokens = []
        current_ids = input_ids

        for _ in range(max_new_tokens):
            outputs = self.model(current_ids)
            next_token_logits = outputs.logits[0, -1, :]

            # Greedy choice
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())

            # Check if we hit newline or EOS
            token_text = self.tokenizer.decode([next_token.item()])
            if '\n' in token_text or next_token.item() == self.tokenizer.eos_token_id:
                break

            # Update input
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return completion

    def complete_multiple_lines(
        self,
        prefix: str,
        num_lines: int = 1,
        max_tokens_per_line: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.95
    ) -> str:
        """
        Complete multiple lines from prefix

        Args:
            prefix: Code prefix
            num_lines: Number of lines to generate
            max_tokens_per_line: Max tokens per line
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        Returns:
            Generated multi-line completion
        """
        current_prefix = prefix
        completions = []

        for _ in range(num_lines):
            line_completion = self.complete_line(
                current_prefix,
                max_new_tokens=max_tokens_per_line,
                temperature=temperature,
                top_p=top_p,
                stop_at_newline=True
            )

            if not line_completion.strip():  # Empty line, stop
                break

            completions.append(line_completion)
            current_prefix = current_prefix + line_completion

            # Stop if we generated EOS
            if self.tokenizer.eos_token in line_completion:
                break

        return ''.join(completions)

    def complete_until_valid_code(
        self,
        prefix: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95
    ) -> str:
        """
        Complete code until a valid stopping point (similar to the generator logic)

        Args:
            prefix: Code prefix
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        Returns:
            Generated completion
        """
        input_ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            completion = self.tokenizer.decode(
                output_ids[0][len(input_ids[0]):],
                skip_special_tokens=True
            )

        # Post-process to remove invalid parts
        completion = self._post_process_completion(completion)

        return completion

    def _post_process_completion(self, completion: str) -> str:
        """
        Post-process completion to remove invalid parts
        (Based on the logic from the original generator)
        """
        # Stop at certain patterns
        stop_pattern = re.compile(r'(Human|Assistant|User|System)')
        match = stop_pattern.search(completion)
        if match:
            completion = completion[:match.start()]

        # Remove consecutive empty lines (4+)
        lines = completion.split('\n')
        valid_lines = []
        consecutive_empty = 0

        for line in lines:
            if line.strip() == '':
                consecutive_empty += 1
                if consecutive_empty < 4:
                    valid_lines.append(line)
            else:
                consecutive_empty = 0
                valid_lines.append(line)

        return '\n'.join(valid_lines).rstrip()

    def batch_complete(
        self,
        prefixes: List[str],
        max_new_tokens: int = 128,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Complete multiple prefixes in batches

        Args:
            prefixes: List of code prefixes
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for processing
            **kwargs: Additional arguments for completion
        Returns:
            List of completions
        """
        completions = []

        for i in range(0, len(prefixes), batch_size):
            batch_prefixes = prefixes[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prefixes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True
            ).to(self.device)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=kwargs.get('temperature', 0.2),
                    top_p=kwargs.get('top_p', 0.95),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode completions
            for j, output in enumerate(output_ids):
                prefix_len = len(inputs.input_ids[j])
                completion = self.tokenizer.decode(
                    output[prefix_len:],
                    skip_special_tokens=True
                )
                completion = self._post_process_completion(completion)
                completions.append(completion)

        return completions
