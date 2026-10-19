# Data licence

The three files in `datasets/` are third-party classification schemes, unmodified apart from the file names.

`datasets/CPCSchemeXML202605.zip` is the [Cooperative Patent Classification](https://www.cooperativepatentclassification.org/) (CPC) scheme in XML, version 2026.05, published jointly by the European Patent Office and the United States Patent and Trademark Office. It is identical to the file they publish at https://www.cooperativepatentclassification.org/sites/default/files/cpc/archive/CPCSchemeXML202605.zip. The CPC website states no licence for this download, so we credit the EPO and USPTO as its source.

`datasets/mesh_descriptors_2026.zip` holds `desc2026.xml`, the [Medical Subject Headings](https://www.nlm.nih.gov/mesh/) (MeSH) 2026 descriptors from the U.S. National Library of Medicine (NLM), downloaded as `desc2026.zip`. Source: U.S. National Library of Medicine, used under the [NLM Terms and Conditions for MeSH](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html). This copy is the MeSH 2026 build dated 13 January 2026 and may not be the most current data available from NLM. NLM has not endorsed this repository or TypeSafe.

`datasets/shopify_categories_2026-02.txt` is `dist/en/categories.txt` from the [Shopify Product Taxonomy](https://github.com/Shopify/product-taxonomy) release v2026-02, licensed under the MIT License, Copyright (c) Shopify. The MIT permission notice is the same as in this repository's [LICENSE](../../LICENSE).

`json_cache.json` contains category names from these three schemes, which stay under the terms above.

Everything else in this folder is TypeSafe's work under the repository's MIT licence.
