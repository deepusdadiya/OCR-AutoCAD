# Generic Floor Plan OCR POC

## What This Project Is

This project is a document-understanding prototype for architectural / AutoCAD-style PDFs. Its job is to turn an uploaded drawing into structured output like:

- room / component name
- area
- page number
- confidence / method metadata
- optional comparison against an expected CSV

From a business point of view, it is trying to replace a manual workflow where someone opens a drawing, reads labels by eye, estimates or copies areas, and then types everything into Excel.

It is not a single AI model. It is a hybrid pipeline:

- PDF text extraction
- OCR fallback
- rule-based label filtering
- vector geometry analysis
- scale detection / inference
- area assignment
- output formatting and comparison

## What Happens When A User Uploads A PDF

1. The user uploads a PDF in the Streamlit app.
   The app can also take an optional expected CSV if the user wants to compare predicted output against a reference sheet.

2. The app saves the uploaded file temporarily and starts the pipeline.
   In Streamlit, this is mainly an in-memory / session workflow for analysis and display. In the CLI flow, outputs are also written to CSV files.

3. The system checks how many pages are in the PDF.
   It then processes the PDF page by page.

4. Each page is rendered as an image.
   This gives the system a visual representation of the page, which is useful for page-region detection and OCR.

5. The system detects the main drawing area versus side metadata.
   It tries to separate the real floor-plan region from title blocks, notes, borders, and side information.
   Business meaning: this reduces noise and helps focus on useful labels.

6. The system extracts text from the PDF.
   First it tries to read native PDF text, which is usually more accurate and faster than OCR.
   If that is not enough, it can retry using OCR.

7. OCR is used only when needed.
   If the PDF text layer is weak or missing, the system tries OCR to read labels from the rendered page image.
   Business meaning: this makes the solution more flexible across different PDF types.

8. The extracted text is cleaned and reconstructed.
   Individual words are grouped into more meaningful text blocks or lines.
   It also normalizes text by fixing spacing, case, and some symbol formatting.

9. The system classifies which text looks like a real label.
   It tries to distinguish:
   - room / component labels
   - dimensions
   - metadata
   - symbols
   - engineering spec text

   It uses heuristics like:
   - whether the text is inside the drawing region
   - font size
   - number of words
   - how alphabetic vs numeric the text is
   - punctuation level
   - whether it looks like a dimension or annotation

10. Only likely room/component labels are kept.
    This is where the pipeline filters out a lot of junk text like dimensions, notes, or title block text.

11. Split labels are fused if needed.
    Sometimes a label appears in multiple parts, such as vertical labels or broken labels across lines.
    The system merges those pieces into one final label.

12. Nearby area annotations are attached if available.
    Example: if a drawing has `PANTRY` on one line and `12.5 SQM` nearby, the system can connect them.

13. The system tries to detect the drawing scale.
    It looks for scale patterns like `1:100`, architectural scale text, or other signals.
    If direct scale is missing, it may infer scale from:
    - embedded area text
    - size annotations near shapes

14. The system rebuilds geometry from the PDF drawing.
    It reads vector linework from the PDF and tries to convert enclosed drawing shapes into polygons.
    Business meaning: this is the key step that lets it calculate area from the plan itself, not just text.

15. Area is assigned to each label through multiple passes.
    The logic is layered because real drawings are messy.

    It first tries the cleanest case:
    - label sits inside a closed polygon

    If that fails, it tries fallback strategies:
    - close small door gaps
    - collect fragmented shapes
    - partition open areas across multiple labels
    - use embedded text area if geometry is not reliable

16. Each resolved area gets a method and confidence.
    This tells us how the area was derived, for example:
    - exact polygon match
    - closed-gap polygon
    - residual open-area partition
    - embedded text area
    - unresolved

17. Very tiny or low-value outputs can be pruned.
    This helps remove obvious noise from the final result.

18. The page-level results are combined into final project outputs.
    The system then builds:
    - text candidates table
    - final labels table
    - debug table
    - optional comparison table if expected CSV is provided

19. If an expected CSV is uploaded, the system compares prediction vs reference.
    Right now the comparison tab shows:
    - `predicted_name`
    - `expected_name`
    - `predicted_area`
    - `expected_area`
    - `Name_Matched`
    - `Area_Matched`

    `Name_Matched` is based on normalized exact-name matching.
    `Area_Matched` is true if predicted area is within 5% of expected area.

20. The app displays everything in Streamlit.
    The user can inspect:
    - page regions
    - text candidates
    - final labels
    - expected vs predicted comparison

## What The User Finally Gets

The most important business output is the structured table of extracted labels and areas.

That means the system turns an unstructured drawing into something closer to operational data:

- names
- measured areas
- quality flags
- comparison against expected output

This is the bridge from "drawing" to "usable business dataset".

## In One Line

"Read the PDF, identify the relevant labels, understand the drawing geometry, estimate areas, and present the result in a structured table."

## Business Value

The value of this system is:

- reducing manual effort
- speeding up validation of room-area sheets
- improving consistency
- giving teams a first-pass automated extraction instead of starting from scratch

This is especially useful where many similar layout PDFs need to be reviewed repeatedly.

## What It Does Well

It works best when:

- the PDF contains readable room/component labels
- the drawing has usable vector geometry
- the scale is present or inferable
- the document is similar to architectural / layout-style floor plans

In those cases, it can produce strong structured output and reduce a lot of manual work.

## What It Does Not Yet Guarantee

It is not a universal PDF engine.

It will struggle when:

- the page has blocks but no labels
- the PDF is only a poor-quality scan
- the geometry is too fragmented
- the scale is missing and cannot be inferred
- the page is a very different drawing type from what the heuristics expect

So the correct product framing is:

- strong for a defined class of PDFs
- not guaranteed for all PDFs

## How I’d Describe It

"This is an intelligent extraction prototype for floor-plan-style PDFs. A user uploads a drawing, the system reads text and geometry together, identifies likely room/component labels, estimates their areas, and returns structured output plus validation against an expected sheet. It already works well on the right class of documents, but it still needs broader validation and confidence-based workflow design before it becomes a production-grade universal solution."
