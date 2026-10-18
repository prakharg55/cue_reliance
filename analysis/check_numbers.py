"""
Checks a paper's LaTeX source against the saved outputs. Every number in the
body must appear in one of results/*.txt, up to rounding. Design constants
such as sample sizes are skipped.

    python analysis/check_numbers.py path/to/paper.tex    # exit code 0 if all are found
"""
import re, sys, glob

if len(sys.argv) < 2:
    sys.exit("usage: python3 check_numbers.py path/to/paper.tex")
TEX = sys.argv[1]
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# structural constants: design grid, sample sizes, counts, years, formatting.
# Listed here so genuinely statistical values stand out in the report.
WHITELIST = {
    11, 1, 2, 3, 4, 5, 8, 9, 10, 16, 20, 25, 50, 15, 18, 32, 36, 72, 96,
    108, 135, 144, 180, 219, 300, 486, 800, 4000, 0.5, 1.5, 1.4, 410,
    0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 0.05, 100, 90, 95,
    2e-4, 2e-5, 0.4,
    425,   # checkpoint identifier OLMo-2-0425-1B, not a statistic
}


def normalize(tex):
    body = tex.split(r"\begin{document}")[1]
    body = re.sub(r"\\bibliography\{[^}]*\}", "", body)
    body = re.sub(r"(?<!\\)%.*", "", body)                       # comments
    body = re.sub(r"\\(?:ref|label|cite[a-z]*)\*?\{[^}]*\}", "", body)
    body = re.sub(r"([\d.]+)\s*\{\\times\}\s*10\^\{(-?\d+)\}", r"\1e\2", body)
    body = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", body)
    body = body.replace(r"\%", " ").replace(r"\times", " ")
    body = body.replace("--", " ")   # en-dash ranges must not read as negatives
    body = re.sub(r"[{}$]|\\[,;!]", "", body)
    return body


def close(a, b):
    if a == 0 or b == 0:
        return abs(a - b) < 1e-9
    if a * b < 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) < 0.03 or abs(a - b) < 5e-4


def pool(paths):
    vals = []
    for p in paths:
        for m in NUM.finditer(open(p).read().replace("−", "-").replace("×", "x")):
            try:
                vals.append(float(m.group()))
            except ValueError:
                pass
    return vals


def main():
    body = normalize(open(TEX).read())
    res = pool(sorted(glob.glob("results/*.txt")))
    seen, nowhere = set(), []
    for line in body.splitlines():
        for m in NUM.finditer(line):
            try:
                v = float(m.group())
            except ValueError:
                continue
            if v in seen or abs(v) in WHITELIST or v.is_integer() and 1900 <= v <= 2030:
                continue
            seen.add(v)
            # percentages render both ways (63% <-> 0.63); accept either scale
            forms = (v, v / 100, v * 100)
            if not any(close(f, x) for f in forms for x in res):
                nowhere.append((v, line.strip()[:70]))
    print(f"{len(seen)} distinct non-structural numbers in tex body")
    print(f"  traceable to results/*.txt : {len(seen)-len(nowhere)}")
    print(f"  NOWHERE (fix these)        : {len(nowhere)}")
    for v, l in nowhere:
        print(f"    {v:g}  |  {l}")
    return 1 if nowhere else 0


if __name__ == "__main__":
    sys.exit(main())
