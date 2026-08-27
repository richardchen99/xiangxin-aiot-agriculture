from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = [['Humidity', 'Normal', '82', 'Sensor signal'], ['Light', 'Watch', '61', 'Sensor signal'], ['Soil moisture', 'Normal', '76', 'Sensor signal'], ['Pest suspicion', 'Review', '18', 'Vision signal'], ['Edge response', 'Improved', '64', 'Latency reduction percent']]


def main() -> None:
    output = ROOT / "data" / "public_safe_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["indicator", "group", "value", "note"])
        writer.writerows(ROWS)
    print(f"Wrote {output.relative_to(ROOT)} with {len(ROWS)} public-safe rows.")


if __name__ == "__main__":
    main()
