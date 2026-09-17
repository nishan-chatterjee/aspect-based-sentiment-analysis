# AspectBench release refresh — 17 September 2026

Release tag: `release-2026-09-17`.

- All seven model families' fine-tuned checkpoint contributions / trained heads
  carry CC BY-NC 4.0 notices. Attribution is required; commercial use requires
  separate permission, subject to the license's exceptions and limitations.
- The fine-tuned SloBERTa checkpoints use CC BY-NC 4.0 with permission from the
  SloBERTa owners, confirmed on 17 September 2026. The original upstream
  SloBERTa release retains its CC BY-SA 4.0 license.
- Current project-owned software is offered under the unmodified PolyForm
  Noncommercial 1.0.0 license. Its explicit institutional-use permissions,
  including educational/public-research use regardless of funding, remain
  intact. Licensing decisions are coordinated by the paper's authors and
  our industrial partner; actual rights-holder authority governs each grant.
- Prominent usage notes appear in the GitHub overview, Hugging Face toolkit
  and all seven model cards. Card-generation code preserves these notices.
- One archived source-specific identifier field is normalized to
  `uuid_source`; historical `uuid_*` input fields remain readable.
- Full article citations are retained across the public release pages.

This is a licensing/documentation refresh, not a new training run. It changes
neither model tensors nor dataset access conditions. Third-party assets retain
their original terms, and earlier MIT grants remain valid. Git history is
preserved. The custom academic software license remains an unadopted review
draft, not an alternative license grant.

Validation: five license-policy tests passed; the archived analysis script
passed Python syntax validation. Published Hugging Face files are checked
against local content without authentication.
