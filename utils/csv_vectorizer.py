import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from utils.text_vectorizer import text_to_vector

_DEFAULT_TEXT_COLUMNS = ("title", "synopsis")


def _select_text_columns(fieldnames: Iterable[str]) -> List[str]:
    field_list = [name for name in fieldnames if name]
    if not field_list:
        return []
    if all(col in field_list for col in _DEFAULT_TEXT_COLUMNS):
        return list(_DEFAULT_TEXT_COLUMNS)
    return field_list


def vectorize_csv(
    csv_path: str,
    text_columns: Sequence[str] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> List[Dict[str, object]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    results: List[Dict[str, object]] = []
    with path.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return results

        columns = list(text_columns) if text_columns else _select_text_columns(reader.fieldnames)
        for row in reader:
            parts = [str(row.get(col, "")).strip() for col in columns]
            combined = " ".join([part for part in parts if part])
            vector = text_to_vector(combined)
            results.append(
                {
                    "row": row,
                    "text": combined,
                    "vector": vector,
                }
            )
    return results


_META_FIELDS = ("title", "url", "genre", "country", "year")


def _vector_dim(results: Sequence[Dict[str, object]]) -> int:
    for item in results:
        vector = item.get("vector")
        if isinstance(vector, list):
            return len(vector)
    return 0


def save_vectors_to_csv(
    results: Sequence[Dict[str, object]],
    output_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> None:
    path = Path(output_path)
    dim = _vector_dim(results)
    vector_fields = [f"v{i}" for i in range(dim)]
    fieldnames = list(_META_FIELDS) + vector_fields
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for item in results:
            row = item.get("row")
            base = {key: "" for key in _META_FIELDS}
            if isinstance(row, dict):
                for key in _META_FIELDS:
                    if key in row:
                        base[key] = row[key]

            vector = item.get("vector", [])
            for idx, value in enumerate(vector):
                if idx < dim:
                    base[f"v{idx}"] = value
            writer.writerow(base)


def save_vectors_to_json(
    results: Sequence[Dict[str, object]],
    output_path: str,
    encoding: str = "utf-8",
) -> None:
    path = Path(output_path)
    with path.open("w", encoding=encoding) as handle:
        json.dump(results, handle, ensure_ascii=False)
