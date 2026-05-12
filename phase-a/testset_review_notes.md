# Testset Review Notes

Manual review sample for 10 questions from `testset_v1.csv`.

1. Question style is mostly faithful to the source documents, but several fallback-generated prompts are generic and should be made more specific if full RAGAS generation is available.
2. Finance questions grounded in `BCTC.md` are answerable because the source contains explicit numeric fields and line items.
3. Privacy questions grounded in `Nghi_dinh_so_13.md` are answerable because the decree text is structured by article and clause.
4. Multi-context prompts are useful for stress-testing retrieval breadth, but some pairs may connect weakly because the fallback generator uses local adjacency.
5. Several reasoning questions are still extractive rather than truly multi-step; these should be upgraded if cost budget allows a second-pass manual curation.
6. Context serialization is stored as JSON text in the CSV so downstream scripts can reconstruct the original list cleanly.
7. The current mix satisfies the required columns and target count, but not all questions have equal difficulty.
8. For production evaluation, at least 10 manually rewritten questions should replace the weakest fallback items.
9. The most reliable questions are those tied to explicit statutory definitions or explicit tax form totals.
10. A future refinement should attach source IDs or article references to each generated row for easier debugging.
