# Data licence

`samples.json` and `json_cache.json` in this directory contain requests from CLINC150.

- **Dataset:** CLINC150, from "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction", EMNLP-IJCNLP 2019,
  https://www.aclweb.org/anthology/D19-1131
- **Authors:** Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew
  Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia
  Tang and Jason Mars
- **Source:** https://github.com/clinc/oos-eval (`data/data_full.json`)
- **Licence:** Creative Commons Attribution 3.0 Unported (CC BY 3.0),
  https://creativecommons.org/licenses/by/3.0/

## Changes made

`samples.json` is a subset of CLINC150, not the full dataset. It holds two samples drawn at
random with a fixed seed from the CLINC150 validation and test sets:

- **`val`**, 400 requests from the validation set: 10 for each of the 30 banking and
  credit-card intents, 50 from the CLINC150 out-of-scope set, and 50 from the other eight
  domains.
- **`test`**, 600 requests from the test set: 15 for each of the 30 banking and credit-card
  intents, 75 from the CLINC150 out-of-scope set, and 75 from the other eight domains.

The request text is unchanged. Requests from the other eight domains are relabeled
`out_of_scope`, and their original intent is kept in the `source` field (for example
`other_domain:weather`). The `intents` list names the 30 banking and credit-card intents.
The cookbook code maps the `out_of_scope` label to the app's `decline` route.

`json_cache.json` holds the recorded routing results of Jev (`jev-1.14.0`) and `gpt-5.5`
for these requests. Each entry is keyed by the request text, so the cache
embeds the CLINC150 text and is covered by the same licence.
