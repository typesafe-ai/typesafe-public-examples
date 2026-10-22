# Data license

`iclr2025_reviews.json` holds 5,727 peer reviews of papers submitted to ICLR 2025,
one review per paper, taken from SNOR v1 (Structured and Normalized OpenReview
Reviews and Comments). Each row holds an `id` (conference, OpenReview paper id,
and review note id), the review `text`, the reviewer's overall `rating` (one of
1, 3, 5, 6, 8 or 10), and the `source_url` of the review on OpenReview. The text
joins three sections of the review form, written as "Summary: ... Strengths: ...
Weaknesses: ...", with runs of whitespace collapsed. The review's questions,
confidence, soundness, presentation and contribution scores, ethics flags, and
the paper's title and authors are not included.

How the rows were chosen, from the 31,160 ICLR 2025 reviews of 7,678 papers in
SNOR: one review per paper was picked with a fixed random seed, and that review
was kept only if its text runs 60 to 450 words (1,737 papers dropped), does not
state its own rating (88 dropped), and does not touch AI-safety-attack topics
(126 dropped, matched case-insensitively on: jailbreak, nsfw, deepfake, toxic,
weapon, wmdp, biosecurity, bioterror, hate speech, violen, malware, pornograph,
sexual, child abuse, csam, terroris, suicide, self-harm, red team, red-team,
prompt injection, harmful content, unsafe content, safety filter, guardrail).
The rows are shuffled with the same seed.

## The reviews

The reviews were written by anonymous ICLR 2025 reviewers and submitted to
OpenReview. The OpenReview Terms of Use (https://openreview.net/legal/terms)
release them under CC BY 4.0:

> By submitting a Comment or Configuration Record (and their accompanying
> Edits) to OpenReview the Submitter agrees that it shall be released to the
> public under the Creative Commons Attribution 4.0 International (CC BY 4.0)
> license.

The same terms define Comments to include "tasked reviews, ... ratings", and
place metadata in the public domain:

> To the extent that the Submitter or OpenReview has a copyright interest in
> metadata associated with a Work, a Creative Commons Public Domain Dedication
> (CC0 1.0) will apply.

Attribution: the ICLR 2025 reviewers, via OpenReview (https://openreview.net),
under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Each row's
`source_url` links to the original review. Changes made: three sections of each
review were joined into one text, reviews were filtered and sampled as described
above, and everything else in the review was left out.

## SNOR v1

The reviews were read from SNOR v1 (Structured and Normalized OpenReview Reviews
and Comments), curated by Mark Neumann and published on Zenodo under CC BY 4.0:
https://doi.org/10.5281/zenodo.15866613. This file is a filtered sample of its
ICLR 2025 reviews, not the dataset itself.

## The cache

`json_cache.json` holds the cached model answers for the cookbook's published
run. Every key is a hash, so the file contains no review text in its keys; the
values hold the proposer's question proposals, Jev's probabilities per question,
and token counts, not the reviews. Everything else in this folder is TypeSafe's
work under the repository's MIT license.
