---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench BGE-M3 dense + MLP

This repository reserves the release location for the HBS and Slovenian
document-level aspect-based sentiment analysis checkpoints for this family.

## Checkpoint status

| Language | Mode | Status | Best validation Macro-F1 |
|---|---|---|---:|
| hbs | masked | Checkpoint file unavailable | 0.8841 |
| hbs | unmasked | Checkpoint file unavailable | 0.8895 |
| slovenian | masked | Checkpoint file unavailable | 0.7514 |
| slovenian | unmasked | Checkpoint file unavailable | 0.7393 |

The validation results survived, but none of the four selected trained MLP-head
files are present in the source tree or model archive. Consequently this
repository currently contains metadata only and cannot be used for inference.
The heads must be recovered or retrained before `masked.pt` and `unmasked.pt`
can be published. No checkpoint from another architecture or mode is used as a
substitute.

Input articles will use the same literal target markup as the other
AspectBench families:

```text
Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala <aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor <aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu.
```

See the shared toolkit at
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
for the available model families and validation tooling.
