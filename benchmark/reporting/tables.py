import os
from typing import List, Dict, Any
import csv


class TableWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def save_table(self, rows: List[Dict[str, Any]], filename: str) -> None:
        if not rows:
            return
        keys = sorted(rows[0].keys())
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def save_table_with_order(self, rows: List[Dict[str, Any]], filename: str, field_order: List[str]) -> None:
        if not rows:
            return
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                # Ensure all fields exist; missing ones become None
                out_row = {k: row.get(k, None) for k in field_order}
                writer.writerow(out_row)


