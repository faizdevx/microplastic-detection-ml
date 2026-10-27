# Updating the Zenodo record

Nothing has been uploaded. Steps for a new version of record 10.5281/zenodo.21073653:

1. On the record page choose "New version". Keep `RAW_DATASET.zip` (unchanged; md5
   `5c5914a238f1fe40babb7d7a5bd1b983`).
2. Run `python tools/checksums.py` and compare the printed table with what Zenodo shows
   after upload.
3. Upload these files:

| File | Why |
|---|---|
| `data/processed/groups.csv` | near-duplicate group of every image |
| `data/processed/splits.json` | hold-out and cross-validation folds |
| `data/processed/scattering_v1.csv` | S1-S4 table produced by `make sim` |
| `data/processed/scattering_v1.config.yaml` | exact configuration of that table |
| `results/data_audit.json` | dataset audit |
| `DATASET_CARD.md`, `MODEL_CARD.md` | documentation |

   `final_dataset.csv` (already on the record) has no documented origin. Either remove it from
   the new version or keep it and say so in the description.
4. Metadata to update:
   - Title and description: describe the archive as 2040 image files in 600 near-duplicate
     groups; describe `scattering_v1.csv` with the provenance section of `DATASET_CARD.md`,
     word for word.
   - Related identifiers: add the GitHub repository (`isSupplementTo`) and the paper once it
     has a DOI.
   - Version: `v2` or the tag chosen for the code release.
   - Licence: CC-BY-4.0 for data (unchanged); the code is Apache-2.0.
5. Put the new version DOI into `RELEASE_NOTES.md` and `CITATION.cff`
   (`TODO-after-upload`).
