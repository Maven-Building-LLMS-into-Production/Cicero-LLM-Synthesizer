# sourcery skip: use-named-expression
import csv
import sys
from pathlib import Path

from .summarizer import Summarizer

P_SCRIPT_DIR = Path(__file__).parent

csv.field_size_limit(sys.maxsize)


HTML_STRINGS = []
with open(P_SCRIPT_DIR / "html_tags.txt") as f:
    for line in f:
        tag = line.strip()
        if tag:
            HTML_STRINGS.append(tag)
HTML_STRINGS.extend(("target_blank", "relnoopen", "relnofollow"))
HTML_STRINGS = tuple(HTML_STRINGS)

WORDS = set()
with open(P_SCRIPT_DIR / "words_alpha.txt") as f:
    for line in f:
        word = line.strip()
        if word:
            WORDS.add(word)


def filter_content(content):
    c_words = content.split()
    c_filt_words = []
    for w in c_words:
        if w not in WORDS:
            while w.startswith(HTML_STRINGS):
                smax = ""
                for s in HTML_STRINGS:
                    if w.startswith(s) and len(s) > len(smax):
                        smax = s
                w = w[len(smax) :]  # noqa: E203
            while w.endswith(HTML_STRINGS):
                smax = ""
                for s in HTML_STRINGS:
                    if w.endswith(s) and len(s) > len(smax):
                        smax = s
                w = w[len(smax) :]  # noqa: E203
        if w:
            c_filt_words.append(w)
    return " ".join(c_filt_words)


def main():
    DF_EMBED_OUT_DICT = {}
    if (P_SCRIPT_DIR / "df_embed_out.csv").exists():
        with open(
            P_SCRIPT_DIR / "df_embed_out.csv",
            encoding="ascii",
            errors="ignore",
        ) as fin:
            for csv_row in csv.DictReader(fin):
                DF_EMBED_OUT_DICT[csv_row["_id"]] = csv_row

    SUMMARIZER = Summarizer()

    with open(
        P_SCRIPT_DIR / "df_embed.csv", encoding="ascii", errors="ignore"
    ) as fin, open(
        P_SCRIPT_DIR / "df_embed_out.csv",
        "w",
        encoding="ascii",
        errors="ignore",
    ) as fout:
        csv_reader = csv.DictReader(fin)
        fieldnames = csv_reader.fieldnames[:]
        fieldnames.append("summary")
        fieldnames.append("content_filtered")
        csv_writer = csv.DictWriter(fout, fieldnames)
        csv_writer.writeheader()
        for csv_row in csv_reader:
            if csv_row["_id"] in DF_EMBED_OUT_DICT:
                print("Re-using existing data for", csv_row["_id"])
                csv_row = DF_EMBED_OUT_DICT[csv_row["_id"]]
            if not csv_row["title"] and not csv_row["content"]:
                csv_row["content"] = csv_row["_id"]
                csv_row["_id"] = ""
            content_filtered = filter_content(csv_row["content"])
            print(content_filtered)
            csv_row["content_filtered"] = content_filtered
            # input()
            if not csv_row.get("summary") and (
                csv_row["title"] or csv_row["content_filtered"]
            ):
                print("Running GPT...\n")
                while True:
                    summary = SUMMARIZER.summarize(
                        csv_row["title"], content_filtered
                    )
                    if summary:
                        break
                # input()
                csv_row["summary"] = summary
            csv_writer.writerow(csv_row)
            fout.flush()


if __name__ == "__main__":
    main()
