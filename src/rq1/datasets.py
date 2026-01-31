"""Dataset loaders for RQ1 experiments (HumanEval + Py150)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.human_eval.human_eval.data import read_problems

LOGGER = logging.getLogger(__name__)


def load_humaneval_dataset(num_samples: Optional[int] = None) -> List[Tuple[str, Dict]]:
    problems = [(task_id, problem) for task_id, problem in read_problems().items()]
    if num_samples:
        problems = problems[:num_samples]
    return problems


def _extract_code_field(record: Dict[str, Any]) -> Optional[str]:
    for key in ("canonical_solution", "code", "content", "source", "text", "snippet", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def load_py150_dataset(
    py150_path: str,
    num_samples: Optional[int] = None,
    py150_min_chars: int = 80,
    py150_max_chars: Optional[int] = 800,
    py150_shuffle: bool = True,
    py150_seed: int = 42,
) -> List[Tuple[str, Dict]]:
    """
    Load Py150 dataset from a directory of .py files or a json/jsonl file.
    Returns a HumanEval-like list[(task_id, {"prompt": str, "canonical_solution": str})].
    """
    dataset_path = Path(py150_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Py150 dataset not found at: {dataset_path}. Provide --py150_path.")

    rng = random.Random(py150_seed)
    samples: List[Tuple[str, Dict]] = []

    def build_problem(record: Dict[str, Any], fallback_id: str) -> Optional[Tuple[str, Dict]]:
        prompt = record.get("prompt", "") or ""
        prompt = prompt if isinstance(prompt, str) else ""
        code_field = _extract_code_field(record)
        if not code_field or not isinstance(code_field, str):
            return None

        canonical_solution = record.get("canonical_solution") or code_field
        if not isinstance(canonical_solution, str):
            return None

        total_text = f"{prompt}{canonical_solution}"
        if len(total_text) < py150_min_chars:
            return None
        if py150_max_chars and len(total_text) > py150_max_chars:
            return None

        task_id = str(record.get("task_id") or record.get("id") or fallback_id)
        return task_id, {"prompt": prompt, "canonical_solution": canonical_solution}

    if dataset_path.is_dir():
        files = sorted(dataset_path.rglob("*.py"))
        if py150_shuffle:
            rng.shuffle(files)

        for idx, file_path in enumerate(files):
            if num_samples and len(samples) >= num_samples:
                break
            code_text = _read_text_file(file_path)
            problem = build_problem({"code": code_text}, f"py150_{file_path.relative_to(dataset_path)}")
            if problem:
                samples.append(problem)

        if num_samples and len(samples) < num_samples:
            LOGGER.warning(
                "Requested %d Py150 samples but only %d were available after filtering",
                num_samples,
                len(samples),
            )

        LOGGER.info("Loaded %d samples from Py150 directory: %s", len(samples), dataset_path)
        return samples

    if dataset_path.suffix.lower() == ".jsonl":
        def iter_jsonl_records():
            with dataset_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line), line_idx
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping malformed JSONL line %d in %s", line_idx + 1, dataset_path)

        if py150_shuffle and num_samples:
            total_valid = 0
            for record, line_idx in iter_jsonl_records():
                problem = build_problem(record, f"py150_{line_idx}")
                if not problem:
                    continue
                total_valid += 1
                if len(samples) < num_samples:
                    samples.append(problem)
                else:
                    replace_idx = rng.randint(0, total_valid - 1)
                    if replace_idx < num_samples:
                        samples[replace_idx] = problem
            if num_samples and len(samples) < num_samples:
                LOGGER.warning(
                    "Requested %d Py150 samples but only %d were available after filtering",
                    num_samples,
                    len(samples),
                )
        else:
            for record, line_idx in iter_jsonl_records():
                problem = build_problem(record, f"py150_{line_idx}")
                if problem:
                    samples.append(problem)
                    if num_samples and len(samples) >= num_samples:
                        break

        LOGGER.info("Loaded %d samples from Py150 JSONL: %s", len(samples), dataset_path)
        return samples

    if dataset_path.suffix.lower() == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("data", [])
        if py150_shuffle:
            rng.shuffle(records)

        for idx, record in enumerate(records):
            if num_samples and len(samples) >= num_samples:
                break
            if not isinstance(record, dict):
                continue
            problem = build_problem(record, f"py150_{idx}")
            if problem:
                samples.append(problem)

        if num_samples and len(samples) < num_samples:
            LOGGER.warning(
                "Requested %d Py150 samples but only %d were available after filtering",
                num_samples,
                len(samples),
            )

        LOGGER.info("Loaded %d samples from Py150 JSON: %s", len(samples), dataset_path)
        return samples

    # TXT list of file paths (e.g., python50k_eval.txt)
    if dataset_path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in dataset_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        base_dir = dataset_path.parent

        def iter_paths():
            for ln in lines:
                candidate = base_dir / ln
                yield candidate

        if py150_shuffle:
            line_indices = list(range(len(lines)))
            rng.shuffle(line_indices)
            iter_indices = line_indices
        else:
            iter_indices = range(len(lines))

        for idx in iter_indices:
            if num_samples and len(samples) >= num_samples:
                break
            candidate = (base_dir / lines[idx]).resolve()
            if not candidate.exists() or candidate.is_dir():
                continue
            code_text = _read_text_file(candidate)
            problem = build_problem({"code": code_text}, f"py150_{candidate.relative_to(base_dir)}")
            if problem:
                samples.append(problem)

        if num_samples and len(samples) < num_samples:
            LOGGER.warning(
                "Requested %d Py150 samples but only %d were available after filtering",
                num_samples,
                len(samples),
            )

        LOGGER.info("Loaded %d samples from Py150 file list: %s", len(samples), dataset_path)
        return samples

    # Fallback: treat as a single plain-text code file
    code_text = _read_text_file(dataset_path)
    problem = build_problem({"code": code_text}, f"py150_{dataset_path.name}")
    if problem:
        samples.append(problem)
    LOGGER.info("Loaded %d samples from Py150 text file: %s", len(samples), dataset_path)
    return samples


def load_rq1_dataset(
    dataset_name: str,
    num_samples: Optional[int] = None,
    py150_path: Optional[str] = None,
    py150_min_chars: int = 80,
    py150_max_chars: Optional[int] = 800,
    py150_shuffle: bool = True,
    py150_seed: int = 42,
) -> List[Tuple[str, Dict]]:
    if dataset_name == "humaneval":
        return load_humaneval_dataset(num_samples)
    if dataset_name == "py150":
        path = py150_path or "data/py150"
        return load_py150_dataset(
            py150_path=path,
            num_samples=num_samples,
            py150_min_chars=py150_min_chars,
            py150_max_chars=py150_max_chars,
            py150_shuffle=py150_shuffle,
            py150_seed=py150_seed,
        )
    if dataset_name == "mbpp":
        raise NotImplementedError("MBPP dataset not yet implemented for RQ1.")

    raise ValueError(f"Unknown dataset: {dataset_name}")
