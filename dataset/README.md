# Dataset Folder – Structure & Usage Guide
This directory provides **evaluation samples** used by FaSTA*’s **inductive-reasoning engine** (`run1.py`) when it validates *new / updated* sub-routine rules.

The learner never **trains** on these images; they are sampled **only at test-time** to decide whether a proposed rule lowers cost without hurting quality.

---

## 1. Directory Layout

```
dataset/
├── object_recoloration/
│   ├── 1.png
│   ├── 1.txt
│   ├── 2.png
│   ├── 2.txt
│   └── …
├── object_replacement/
│   ├── 1.png
│   ├── 1.txt
│   └── …
├── object_removal/
│   └── …
├── text_removal/
│   └── …
└── text_replacement/
    └── …
```

* **Folder name = sub-task** (lower-case, words joined by “_”).  
  Must match the `subtask` field in `subroutine_rule_table.json`  
  (`"Object Recoloration"` → `object_recoloration`, etc.).
* **Image file**:  _n_.png  – RGB PNG or JPEG (any resolution).  
* **Prompt file**: _n_.txt – *single line* natural-language instruction
  that **concerns only this sub-task**.  
  The prompt should not request additional subtasks (those would bias
  the cost/quality check).

> **Tip:**  At least **25 pairs** per folder are recommended; `run1.py`
> draws 25 random samples each time it evaluates a rule (or the maximum
> available if fewer).

---

## 2. Prompt Conventions

| Sub-task folder          | Prompt examples                                              |
|--------------------------|--------------------------------------------------------------|
| `object_recoloration`    | “Recolor the **lighthouse** to pink.”                        |
| `object_replacement`     | “Replace the **cat** with a **rabbit**.”                     |
| `object_removal`         | “Remove the **bench** from the scene.”                       |
| `text_removal`           | “Erase the word **‘STOP’** from the sign.”                   |
| `text_replacement`       | “Change **‘CLOSED’** to **‘OPEN’** on the door sign.”        |

*Mention only one primary operation per prompt.*  
If several objects need editing, create separate samples.

---

## 3. File Naming Rules
* Numbers must be **consecutive** within each folder (1, 2, 3 …).  
  Gaps are ignored but help avoid name collisions.
* The *n*.png ↔ *n*.txt pair **must** share the same number.

---

## 4. Adding New Sub-task Types
1. Create a folder whose name is the sub-task in **snake_case**  
   (e.g. `depth_estimation`).
2. Add ≥25 `(image, prompt)` pairs following the numbering scheme.
3. No code changes are required – the learner discovers the folder
   automatically.

---

## 5. Common Pitfalls
| Issue                                   | Fix                                                            |
|-----------------------------------------|----------------------------------------------------------------|
| Folder name does **not** match sub-task | Rename folder to snake-case version of the exact sub-task name |
| Missing .txt file                       | Add prompt file or remove orphan .png                          |
| Prompt contains extra subtasks          | Split into separate sample pairs                               |
| Less than 25 samples                    | Possible, but statistical power drops for the net-benefit test |

---

## 6. Quick Checklist
- [ ] One folder **per** supported sub-task
- [ ] Numbered `n.png` / `n.txt` pairs
- [ ] ≥ 25 valid pairs per folder
- [ ] Prompts contain **only** that folder’s sub-task
- [ ] Images are PG or PNG, RGB

Keep this structure and FaSTA* will automatically perform reliable,
reproducible rule verification after every 20 execution traces.
