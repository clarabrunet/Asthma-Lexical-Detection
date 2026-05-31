# `data/` — clinical corpus (not distributed)

The clinical data used in the thesis are **confidential** and are **not** part
of this repository. This folder is intentionally empty (everything except this
file is git-ignored). To run the pipeline, place your dataset here (or point the
`ASTHMA_E1_DATA` environment variable at another folder).

## Expected layout

```
data/
└── dataset/
    ├── asma/                     # active-asthma patients (label = 1)
    │   ├── <patient_id>/
    │   │   ├── note_1.txt
    │   │   └── note_2.txt
    │   └── ...
    └── no_asma/                  # non-asthma patients (label = 0)
        ├── <patient_id>/
        │   └── note_1.txt
        └── ...
```

- Each subfolder under `asma/` or `no_asma/` is one patient; the folder name is
  the patient id.
- Each patient folder holds the patient's clinical notes as `.txt` files.
- Lab / microbiology reports (file names ending in `_labor`, `_micro`, `_lab`,
  `_microb`, `n2_labor`) are automatically excluded during loading.
- Encoding is detected automatically (utf-8 / cp1252 / latin1).

The class of each patient must be reflected by the folder (`asma` vs `no_asma`)
before running the pipeline.
