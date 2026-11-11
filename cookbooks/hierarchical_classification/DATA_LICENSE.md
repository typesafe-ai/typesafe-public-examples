# Data licence

`datasets/mesh_descriptors_2026.zip` holds the [Medical Subject Headings](https://www.nlm.nih.gov/mesh/) (MeSH) 2026 descriptors, build of 13 January 2026, from the U.S. National Library of Medicine, unmodified apart from its name. It is used under the [NLM Terms and Conditions for MeSH](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html). It is not the most current MeSH data.

The code downloads the [Cooperative Patent Classification](https://www.cooperativepatentclassification.org/) scheme, version 2026.05, from the European Patent Office and the United States Patent and Trademark Office, and the [Shopify Product Taxonomy](https://github.com/Shopify/product-taxonomy) release v2026-02 (`dist/en/categories.txt`, Copyright (c) Shopify, MIT License) from their publishers.

`hierarchical_classification_cache.json` contains category names from these three schemes, which stay under their publishers' terms.

The other files this cookbook publishes to typesafe-public-examples are TypeSafe's work under that repository's MIT licence.
