import json
import sys

def is_noise(entry):
    if not entry: return True
    text = (entry.get("content") or "").strip()
    if not text: return True
    if text in {"{", "}", ";"}: return True
    low = text.lower()
    if "traceback" in low or "object has no attribute" in low:
        return True
    return False

def main(src, dst):
    kept = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if is_noise(obj):
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Cleaned dataset written to {dst} (kept {kept} lines)")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "chris sunny_dataset.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "chris_sunny_dataset.cleaned.jsonl"
    main(src, dst)
