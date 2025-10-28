import csv
import sys

def extract_hierarchy(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8', errors='ignore') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write header for the new CSV
        writer.writerow(["NoticeId", "Department", "SubTier", "Office", "Title"])

        for row in reader:
            nid = (row.get("NoticeId") or "").strip()
            level1 = (row.get("Department/Ind.Agency") or "").strip()
            level2 = (row.get("Sub-Tier") or "").strip()
            level3 = (row.get("Office") or "").strip()
            title = (row.get("Title") or "").strip()

            if nid:  # only write rows with a NoticeId
                writer.writerow([nid, level1, level2, level3, title])

    print(f"Extracted hierarchy to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_hierarchy.py input.csv output.csv")
        sys.exit(1)

    _, input_csv, output_csv = sys.argv
    extract_hierarchy(input_csv, output_csv)

