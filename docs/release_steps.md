# Release steps

None of these commands has been run. Run them in order from a clean checkout of `main` after the
release branch is merged.

## 1. Check

```bash
python tools/release_check.py
```

```bash
python tools/make_bundle.py
```

`results/release_check.md` must end with `Result: passed`. The bundle is written to
`release_bundle/` with `CHECKSUMS.txt`.

## 2. Tag

```bash
git tag -a v0.3.0 -m "v0.3.0: YOLOv11-seg controls, paired comparison, mask review set"
```

```bash
git push origin v0.3.0
```

## 3. GitHub release

```bash
gh release create v0.3.0 --title "v0.3.0" --notes-file RELEASE_NOTES.md release_bundle/CHECKSUMS.txt release_bundle/BUNDLE_INFO.json
```

## 4. Zenodo

Follow `docs/zenodo_update.md`. Upload the files in its table, then compare the checksums Zenodo
shows with `release_bundle/CHECKSUMS.txt`. Put the new DOI into `RELEASE_NOTES.md` and
`CITATION.cff` in place of `TODO-after-upload`, commit, and push.
