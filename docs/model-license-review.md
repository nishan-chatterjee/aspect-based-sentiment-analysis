# Model and software license review — 2026-09-16

The requested policy is noncommercial model reuse with attribution. The six
non-Slavic-specific families and HBS BERTić AspectBench checkpoint contributions
now carry CC BY-NC 4.0 notices. Upstream assets retain their original terms;
the frozen BGE encoder remains MIT, separate from the noncommercial MLP heads.
All eight Hugging Face cards cite the Frontiers article and plain abstract URL.

## Outstanding rights-holder decisions

1. Slovenian SloBERTa derives from EMBEDDIA/sloberta, whose public model card
   declares CC BY-SA 4.0. ShareAlike is not compatible with simply imposing an
   NC restriction. Obtain a separate upstream permission, replace this base
   model, or accept compatible ShareAlike terms (commercial reuse permitted).
   No new license grant is applied to the Slovenian checkpoints in this pass.
   Review their current public distribution promptly; a blanket noncommercial
   model-release statement is not accurate while this remains unresolved.
2. Software currently has an MIT license, including its previous public
   releases. MIT, Apache-2.0 and GPL permit commercial use; GPL adds copyleft,
   not a noncommercial restriction. CC advises against its licenses for
   software. BSL 1.1 changes to a compatible open-source license no later than
   four years after a version's first public release, so does not provide a
   permanent noncommercial policy. A software-specific research/noncommercial
   license, potentially with separate commercial licensing, needs agreement
   by the relevant rights holders and institutional/legal review before use.
   The TurboGAP Academic Software Licence is a project-specific candidate,
   not a universal standard to adopt without reviewing its complete terms.
3. Noncommercial is purpose-based, not a ban on all companies or an automatic
   exception for all academic work. Do not describe CC BY-NC as academic-only.
4. New notices cannot revoke previously granted rights or third-party rights.
   Review historical distributions and ownership before making exclusivity
   claims. Model-weight protection/enforceability itself can depend on the
   jurisdiction; licensing is not a guaranteed technical prevention of reuse.

## Primary references

- [Creative Commons FAQ: software and noncommercial scope](https://creativecommons.org/faq/)
- [CC BY-NC 4.0 legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
- [CC BY-SA 4.0 legal code, sections 2(a)(5) and 3(b)](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
- [SloBERTa upstream model card](https://huggingface.co/EMBEDDIA/sloberta/blob/main/README.md)
- [Business Source License 1.1](https://mariadb.com/bsl11/)
- [TurboGAP Academic Software Licence](https://turbogap.fi/wiki/index.php/Academic_Software_Licence)

Base model license metadata checked anonymously on Hugging Face on 2026-09-16:
XLM-R MIT; XLM-R Longformer Apache-2.0; mDeBERTa-v3 MIT; mT5 Apache-2.0;
BERTić Apache-2.0; SloBERTa CC BY-SA 4.0; BGE-M3 MIT. This metadata check is
not a complete legal audit of provenance or historical upstream revisions.
