# Object boxes

OWLv2 (`google/owlv2-base-patch16-ensemble`) boxes for the figure and ground noun of every item used in the paper. Releasing them makes every result independent of rerunning the detector.

| File | Items | Keys |
|---|---|---|
| `whatsup_owlv2-base-patch16-ensemble.json` | 820 What's Up photographs (subsets A, B) | `A/<file>.jpeg`, `B/<file>.jpeg` |
| `vsr_owlv2.json` | 969 VSR captions (random split, dev + test) | `dev/<line>`, `test/<line>` |
| `e3_owlv2.json` | 2,313 hard negatives (1,136 SugarCrepe replace-relation, 1,177 ARO VG-Relation) | `replace_rel/<index>`, `vg_relation/<index>` |

## Format

```json
{
  "A/beer-bottle_on_armchair.jpeg": {
    "size": [width, height],
    "boxes": {"beer bottle": {"box": [x0, y0, x1, y1], "score": 0.64},
              "armchair":    {"box": [x0, y0, x1, y1], "score": 0.74}},
    "flags": []
  }
}
```

Boxes are in original image pixels. For each noun the detector's best candidate is kept. If both nouns land on the same box (IoU > 0.7), the weaker noun takes its best non-overlapping candidate and is flagged `reassigned:<noun>`, or the item is flagged `same_box` if no such candidate exists. A noun whose score is below 0.2 is flagged `low_score:<noun>`.

| Data | Items | Flagged |
|---|---|---|
| What's Up | 820 | 48 (5.9%) |
| VSR | 969 | 305 (31.5%) |
| Hard negatives | 2,313 | 740 (32.0%) |

The paper reports every result on unflagged items as well; they match the full sets. A visual check of 30 random What's Up images found both boxes on the named objects in all 30, including the four flagged ones.

## Licence

The boxes are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are annotations of images that remain under their own datasets' licences (What's Up, VSR/COCO, SugarCrepe/COCO, ARO/Visual Genome). The images themselves are not included; the loaders fetch them from their official sources.
