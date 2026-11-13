# Contributing

Place each paper in one canonical topic section of `README.md`. Cross-link to that section when another topic needs it.

- Use a primary DOI, arXiv abstract page or official model/project source. Match the title, authors, year and archival venue to the linked version.
- Label a preprint as a preprint unless formal publication is verified. Keep model cards, demonstrations and quantitative studies distinguishable.
- Explain the deployment role in one sentence. Include relevant hardware/workload limits instead of a context-free speed or energy ranking.
- Link code from the paper or publisher/project documentation. A live repository alone does not establish official ownership.
- Prefer active deployment paths. Remove superseded duplicates, placeholders and unsupported artifact associations; foundational studies may remain when their role is clear.
- Keep figures editable and include their source. Quantitative panels need traceable data; schematic geometry must not imply measurements.

## Check a change

```sh
python3 survey/check_docs.py
python3 survey/verify_corpus.py --offline
python3 survey/verify_repos.py
```

The offline literature check reuses cached metadata; a new identifier needs a live check with `verify_corpus.py` and human inspection of the primary record. Review every title-similarity flag and record version renames or deliberate shorthand rather than silently excluding a valid paper. Commit current manifests with the curated change. Do not replace the pinned census used by the manuscript.
