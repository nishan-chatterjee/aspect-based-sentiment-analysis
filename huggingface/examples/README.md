# Tagged inference examples

These two files exercise the public input contract:

```json
{
  "records": [
    {
      "article": "... <aspect>target mention</aspect> ...",
      "sentiment": 1
    }
  ]
}
```

`article` is required and must contain literal `<aspect>...</aspect>` tags.
`sentiment` is optional gold data using `-1` (negative), `0` (neutral), or `1`
(positive). Additional metadata fields such as `notice` and `variety` are
ignored by inference. Tag only the target to classify. If that target occurs
repeatedly or appears in an inflected form, tag every intended mention:

```text
Podjetje <aspect>Modri Gaj</aspect> je objavilo poročilo. Podpora
<aspect>Modrega Gaja</aspect> je odgovorila še isti dan.
```

- `hbs-tagged-examples.json` contains ten machine-generated paragraphs: six
  broadly Serbo-Croatian examples, three Croatian examples, and one Bosnian
  example.
- `sl-tagged-synthetic-examples.json` contains ten machine-generated Slovenian
  paragraphs.

These files exist only for quick single/batch inference checks. They are not
copied or adapted from AspectBench, CLARIN, or a private dataset; they do not
represent the training or release-data distribution and must not be used for
model evaluation or reported performance comparisons.

The masked models replace each tagged span—including its literal text—with
`[ASPECT]` before tokenization. The tags therefore identify *where* the target
appears without disclosing its name to the classifier.
