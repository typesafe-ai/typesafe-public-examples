# Data licence

`candidate_pairs.json` pairs records from Wikidata and MusicBrainz. Each pair joins an album as described by two catalogues. Both sources release
this data under CC0 1.0 (public domain dedication), so no attribution is
required. We credit them anyway so readers can trace a record back.

- **`entity_a`: Wikidata.** Items that are an instance of *album* (Q482994)
  and carry a MusicBrainz release group ID (P436), with title, performer
  (P175), publication date (P577) and record label (P264). Structured data on
  Wikidata is released under CC0 1.0: https://www.wikidata.org/wiki/Wikidata:Licensing
- **`entity_b`: MusicBrainz.** One release from the linked release group,
  with title, artist credit, release date and label, fetched from the
  MusicBrainz web service. For known matches we picked the release whose title
  differs most from Wikidata's, so reissues, localised titles and alternate
  artist credits are over-represented on purpose. MusicBrainz core data is CC0 1.0:
  https://musicbrainz.org/doc/MusicBrainz_Database

`known_same_as` is true when the Wikidata item links to the MusicBrainz
release group the entity_b release belongs to. Negative pairs take a different
MusicBrainz release group with a similar title.

Fetched on 2026-10-16. The file is a snapshot; both sources are community
edited and may have changed since. Titles, artists and labels were filtered
against a word list to keep the file family-friendly, and the 450 pairs were
sampled with a fixed seed. Known matches where both catalogues agree on title
and artist were capped at eight so the set exercises judgment rather than
string equality.

Everything else in this folder is TypeSafe's work under the repository's MIT licence.
