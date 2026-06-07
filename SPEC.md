# Scholar Lens - Session 1 Spec

## Project Goal
Build a clean, professional Gradio application called "Scholar Lens" — a universal research assistant.

## Core Requirements

### App Structure (Gradio Blocks)
- Dark professional theme (#0f172a background, #3b82f6 blue accent)
- Header: "Scholar Lens" with subtitle "Search 200M+ papers across Semantic Scholar, arXiv & PubMed"
- Three tabs:
  1. **Search** (main tab)
  2. **Summarize**
  3. **About**

### Tab 1: Search
- One search input: "Search any research topic..."
- Button: "Search All Sources"
- Show loading state while fetching
- Results displayed in a clean table with these columns:
  - Title
  - Year
  - Source (with colored badges)
  - Authors (shortened)
  - Citations
- Each row should be clickable (for future detail view)

### General Rules
- Use type hints on all functions
- Add proper error handling with user-friendly messages
- Create a clean `requirements.txt` with pinned versions
- Make the UI look modern and professional (not default Gradio look)
- Add basic CSS styling for better appearance

## Output Files
- Create `app.py` with the full Gradio application
- Create `requirements.txt`
- Keep the code clean and well-commented