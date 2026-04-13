import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "arxiv_metadata.csv"
OUTPUT_SQL = BASE_DIR / "sql_script" / "inserts_arxiv_metadata.sql"
BATCH_SIZE = 250


def sql_literal(value: str | None) -> str:
    if value is None or value == r"\N" or value == "":
        return "NULL"

    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace("'", "''")
        .replace("\x00", "")
    )
    return f"'{escaped}'"


def main() -> None:
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as infile, OUTPUT_SQL.open(
        "w", encoding="utf-8", newline="\n"
    ) as outfile:
        reader = csv.DictReader( # following CSV export in 01_EDA_and_processing.ipynb
            infile,
            escapechar="\\",
            quotechar='"',
        )


        outfile.write("SET NAMES utf8mb4;\n")
        outfile.write("START TRANSACTION;\n\n")

        batch: list[str] = []
        for row in reader:
            values_sql = ", ".join(
                [
                    sql_literal(row["id"]),
                    sql_literal(row["paper_id"]),
                    sql_literal(row["title"]),
                    sql_literal(row["authors"]),
                    sql_literal(row["categories"]),
                    sql_literal(row["update_year"]),
                ]
            )
            batch.append(f"({values_sql})")

            if len(batch) >= BATCH_SIZE:
                write_batch(outfile, batch)
                batch.clear()

        if batch:
            write_batch(outfile, batch)

        outfile.write("COMMIT;\n")

    print(f"Wrote {OUTPUT_SQL}")


def write_batch(outfile, batch: list[str]) -> None:
    outfile.write(
        "INSERT INTO papers (id, paper_id, title, authors, categories, update_year)\n"
    )
    outfile.write("VALUES\n")
    outfile.write(",\n".join(batch))
    outfile.write(
        "\nON DUPLICATE KEY UPDATE\n"
        "    paper_id = VALUES(paper_id),\n"
        "    title = VALUES(title),\n"
        "    authors = VALUES(authors),\n"
        "    categories = VALUES(categories),\n"
        "    update_year = VALUES(update_year);\n\n"
    )


if __name__ == "__main__":
    main()
