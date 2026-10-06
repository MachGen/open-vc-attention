"""Recompute paired statistics from completed report-reproduction results."""

import argparse
import json

from common import validate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    print("| Shape | Route | Median ms |")
    print("|---|---|---:|")
    for path in args.results:
        report = json.loads(open(path).read())
        result = validate_result(report)
        for route, median in result["median_ms"].items():
            print(f"| {'x'.join(map(str, report['shape']))} | {route} | {median:.6f} |")
        print("\n" + json.dumps(result["comparisons"], indent=2))
        if not report["isolation_checked"]:
            print("Shared GPU: correctness/runner smoke only; not an isolated performance claim.")


if __name__ == "__main__":
    main()
