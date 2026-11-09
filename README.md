# Dataset

Place the supplied `stressed vs normal(Sheet1).csv` here for local training,
or upload it when the Colab notebook requests a CSV.

Required columns: `post` (text) and `label` (`normal` or `stressed`).
The supplied original has 1,800 rows, 900 per class, and decodes with cp1252.
Its SHA-256 is `2298216368aec97b4c1ccc346ac9e7f6e59a9bbdefcf4dd22056dffd32fdf891`.

The data was supplied for this assignment. An original source URL, license,
annotation protocol and author/thread identifiers were not provided. Do not claim
that it is a named public dataset without verifying its provenance. The raw CSV is
excluded from this GitHub-ready package. Add a proper dataset citation and permitted
access instructions once its original source is known.

Cleaning removes two rows in one conflicting-label group and two redundant
same-label copies, leaving 1,796 records (897 normal and 899 stressed).
`artifacts/split_manifest.csv` maps retained zero-based original record indices to
the fixed train/validation/test split. CSV multiline records mean record indices
are not physical line numbers.
