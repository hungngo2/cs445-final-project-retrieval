# OfficeQA Page Retrieval with Sparse Text and Multimodal Indexing

## Motivation and Impact

Document question answering systems are only useful if they can first retrieve the right evidence page. OfficeQA is a difficult testbed for this problem because its source documents are long Treasury Bulletin PDFs with dense tables, repeated layouts, and many visually similar pages. A successful retriever must handle both lexical specificity, such as years and financial terms, and visual structure, such as tables and section boundaries.

We chose this project to test whether multimodal retrieval models can act as strong first-stage retrievers for document QA, or whether a simpler sparse text baseline remains more reliable. This matters more broadly for document understanding systems over reports, scanned archives, and other layout-heavy material. If multimodal retrieval works well, it could reduce dependence on OCR or brittle text extraction. If it does not, that result is still useful because it clarifies where sparse text methods remain hard to beat.

## Approach

We implemented a retrieval-only OfficeQA benchmark centered on page retrieval. The pipeline first downloads the Treasury Bulletin PDFs referenced by the OfficeQA CSV and builds a page-level manifest. Each manifest entry stores the document ID, page number, page dimensions, and page text extracted with PyMuPDF.

Our text baselines are BM25 over native PDF text and BM25 over OCR text. For native BM25, we index the text extracted directly from the PDF with PyMuPDF. For OCR BM25, we run PaddleOCR on rendered pages and index the resulting page text with the same BM25 pipeline. These baselines are strong because many OfficeQA questions include exact years, named entities, and financial phrases.

Our multimodal baselines use CLIP and SigLIP text-image embeddings indexed with FAISS. We evaluate three page representations. In the full-page setting, the entire rendered page is embedded as one image. In the fixed-crop setting, each page is split into a deterministic 2x2 grid and each crop is indexed separately; the page score is the maximum crop score for that page. In the layout-aware setting, we use PDF-native structure cues from PyMuPDF to propose semantically meaningful regions such as tables, image blocks, and merged text blocks, then again aggregate region scores back to the page level.

This gives an apples-to-apples comparison: all methods retrieve pages from the same corpus, use the same gold labels, and are evaluated with the same metrics. We evaluate on the full OfficeQA Pro benchmark used in our experiments, containing 133 questions and a page manifest of 24,303 pages. For each method, we retrieve the top 50 pages per query and report page Recall@1, Recall@5, Recall@10, page MRR, and nDCG@10. We also compute document-level metrics by collapsing page rankings to unique documents.

![Figure A: System architecture for shared sparse and multimodal retrieval.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/architecture_diagram.svg)

*Figure A. System architecture. The same OfficeQA questions and Treasury Bulletin page corpus feed both retrieval branches: a sparse BM25 page index and a multimodal FAISS index family built from full pages, fixed crops, or layout-aware regions.*

![Figure B: End-to-end experiment pipeline.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/pipeline_diagram.svg)

*Figure B. End-to-end experiment pipeline. Expensive preprocessing and indexing are separated from cheap repeatable evaluation, which lets us rerun retrieval experiments and analysis from saved indexes without rebuilding the entire corpus.*

## Results

The clearest result is that text retrieval is the strongest first-stage strategy on this benchmark. Table 1 summarizes the main numbers from the final full run.

| Run | Page MRR | Page R@10 | Doc MRR | Doc R@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.0572 | 0.0827 | 0.1776 | 0.3383 |
| BM25 OCR | 0.0498 | 0.0827 | 0.1808 | 0.3308 |
| CLIP fixed 2x2 | 0.0225 | 0.0451 | 0.0549 | 0.1278 |
| CLIP layout-aware | 0.0200 | 0.0451 | 0.0695 | 0.1579 |
| SigLIP fixed 2x2 | 0.0070 | 0.0301 | 0.0866 | 0.2180 |
| SigLIP layout-aware | 0.0092 | 0.0150 | 0.0484 | 0.1203 |
| CLIP full | 0.0017 | 0.0075 | 0.0663 | 0.1504 |
| SigLIP full | 0.0007 | 0.0000 | 0.0584 | 0.1353 |

Native BM25 is the best page-level retriever, with page MRR 0.0572. OCR BM25 is slightly lower on page MRR at 0.0498, but it ties native BM25 on page Recall@10 and slightly improves document MRR from 0.1776 to 0.1808. This suggests OCR changes the ranking of some documents usefully, but does not improve exact page localization overall. The best multimodal page retriever is CLIP with fixed crops, but its page MRR of 0.0225 is still well below both text baselines.

![Figure 1: Page-level retrieval metrics across all runs.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/summary_page_metrics.png)

*Figure 1. Page-level retrieval summary on the full OfficeQA benchmark. Native BM25 and OCR BM25 are the strongest systems. Fixed-crop and layout-aware multimodal variants clearly outperform full-page multimodal retrieval, but still trail the text baselines.*

![Figure 2: Document-level retrieval metrics across all runs.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/summary_doc_metrics.png)

*Figure 2. Document-level retrieval summary. OCR BM25 slightly improves document MRR over native BM25, while native BM25 still has the best document Recall@10. This helps separate coarse document retrieval from exact page localization.*

The most useful ablation result is that localization matters a great deal. Full-page multimodal retrieval is extremely weak: CLIP full reaches only 0.0017 page MRR, while SigLIP full reaches 0.0007. Both fixed crops and layout-aware regions improve performance substantially. For CLIP, fixed crops improve page MRR by more than an order of magnitude relative to full-page retrieval, and layout-aware crops provide a similar gain. SigLIP shows the same pattern, although its absolute page-level performance remains lower than CLIP on this task.

This means our negative result is still informative. Multimodal retrieval is not useless here, but whole-page single-vector representations are too coarse. Region-level retrieval is clearly better, yet sparse text matching remains more effective for this benchmark.

![Figure 3: Localization ablation relative to full-page retrieval.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/crop_benefit.png)

*Figure 3. Localization benefit for multimodal retrieval. Both fixed 2x2 crops and layout-aware regions improve over full-page embeddings, showing that smaller evidence regions are much better aligned with page retrieval than a single whole-page image vector.*

Another important pattern is the gap between document-level and page-level performance. Even for native BM25, document Recall@10 is 0.3383 while page Recall@10 is only 0.0827. OCR BM25 shows a similar pattern: document Recall@10 is 0.3308 while page Recall@10 remains 0.0827. This suggests that the benchmark is hard not only because relevant documents are difficult to find, but because many questions require precise page grounding within long reports. In other words, page localization is the main bottleneck.

![Figure 4: Recall@k curves for page and document retrieval.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/recall_curves.png)

*Figure 4. Recall@k curves further emphasize the same trend: text retrieval improves more quickly as k increases, while localized multimodal retrieval helps relative to full-page retrieval but remains below the sparse baselines.*

![Figure 5: Document-vs-page retrieval gap.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/results_new/doc_page_gap.png)

*Figure 5. Document-vs-page retrieval gap. All methods retrieve correct documents more often than exact evidence pages, showing that fine-grained page localization is the harder part of the benchmark.*

Our qualitative analysis supports this interpretation. BM25 often succeeds when a query contains strong lexical anchors such as a year, month, statistic, or Treasury-specific term. The multimodal systems often retrieve visually similar tables or layouts from the wrong bulletin or wrong year. Localization reduces this problem by narrowing attention to smaller regions, but it does not fully solve the semantic mismatch.

![Figure 6: Example query gallery.](/Users/hungvinhngo/workplace/cs445-final-project/report_assets/query_galleries/UID0001_query_gallery.png)

*Figure 6. Qualitative query gallery. The gallery compares gold pages against retrieved pages from several runs. These examples help diagnose whether failures are due to wrong documents, wrong pages within a document, or visually similar but semantically mismatched tables.*

## Implementation Details

We implemented the project in Python. Native PDF parsing and rendering use PyMuPDF, crop generation uses Pillow, sparse retrieval uses rank-bm25, OCR uses PaddleOCR, multimodal encoders use Hugging Face Transformers with PyTorch, and vector search uses FAISS. Heavy indexing and evaluation were run in Google Colab Pro because the full benchmark requires rendering and embedding tens of thousands of page images.

The pipeline has four stages: build a page manifest from the referenced PDFs, create BM25 indexes over native and OCR page text, build CLIP and SigLIP FAISS indexes for full-page and cropped page representations, and run retrieval evaluation with exported analysis tables and plots.

One implementation decision worth noting is that our layout-aware retrieval is not based on a learned layout detector such as PaddleOCR or LayoutLM. Instead, we use PDF-native structure cues from PyMuPDF, including table detection, image blocks, and merged text boxes. We chose this approach because it gives a meaningful structure-aware baseline while staying lightweight enough to run at full-dataset scale. PaddleOCR is used separately only for the OCR BM25 text baseline.

We also built experiment infrastructure to make the benchmark practical: incremental index construction, reusable render caches, progress monitoring, per-experiment analysis exports, query galleries, and summary visualizations. These choices mattered because the full benchmark contains 24,303 pages and the fixed-crop condition expands this to 97,212 indexed image regions.

External resources used in this project include the OfficeQA-style question CSV, the Treasury Bulletin PDFs downloaded from FRASER, PaddleOCR, the open-source CLIP and SigLIP checkpoints from Hugging Face Transformers, FAISS for vector search, and Google Colab for large-scale execution.

## Challenge and Innovation

The most challenging part of the project was not implementing a single retriever, but designing a controlled benchmark across very different retrieval paradigms. Sparse text retrieval, OCR-based text retrieval, full-page multimodal retrieval, fixed-grid crops, and layout-aware region crops all place different demands on preprocessing, indexing, runtime, and analysis. Making these methods comparable required careful engineering around manifests, cache reuse, batching, page-to-document aggregation, and unified evaluation outputs.

Another challenge was that the outcome was genuinely uncertain. It was plausible that multimodal models would outperform BM25 on layout-heavy documents, especially with localized crops. Our final results showed something more nuanced: localization helps substantially, but multimodal first-stage retrieval still does not beat BM25. We believe that is a useful finding because it clarifies where multimodal retrieval is promising and where sparse lexical methods remain strong.

We also went beyond a single baseline comparison by implementing native BM25, OCR BM25, fixed and layout-aware localization strategies, plus a full analysis workflow with recall curves, win-loss summaries, qualitative galleries, and first-hit rank analysis. Based on the course rubric, we believe this project merits **18/20** for challenge and innovation. It goes beyond a basic technique, includes multiple nontrivial ablations, and required overcoming practical scaling and evaluation issues across text, OCR, and multimodal retrieval.

## Conclusion

Native BM25 remains the strongest first-stage page retriever for OfficeQA-style financial documents. OCR BM25 is competitive and slightly improves document MRR, but it does not improve exact page MRR. Full-page multimodal retrieval with CLIP and SigLIP performs poorly, while localization through fixed crops or layout-aware regions improves multimodal retrieval substantially. Even with those gains, multimodal first-stage retrieval does not surpass sparse text retrieval on the full benchmark.

This suggests two promising future directions: hybrid text retrieval that combines native extraction and OCR more selectively, and multimodal models used as rerankers or fine-grained evidence localizers rather than standalone first-stage retrievers.

## Group Contributions

Replace this section with your actual team breakdown before submitting. A simple format that should work well is:

- Member 1: data pipeline, BM25 baseline, evaluation code
- Member 2: CLIP/SigLIP multimodal indexing, FAISS retrieval
- Member 3: OCR BM25 baseline, layout-aware crops, analysis notebook
- Member 4: report writing, experiment orchestration, debugging and validation

## References

- OfficeQA dataset and assignment-provided question CSV
- FRASER Treasury Bulletin PDF collection
- PaddleOCR
- CLIP: Radford et al.
- SigLIP: Zhai et al.
- FAISS: Johnson et al.
