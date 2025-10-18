from typing import List, Dict, Any


def build_context_from_search_results(search_results: List[Dict[str, Any]]) -> str:
    """
    Build formatted context string from search results for LLM prompt.

    Transforms search results into a formatted text block with source metadata
    and content, ready to be included in an LLM prompt.

    Args:
        search_results: List of search result dictionaries with structure:
            {
                "content": str,
                "metadata": {
                    "doc_id": str,
                    "doc_title": str,
                    "section_title": str,
                    "section_path": str,
                    "page_start": int,
                    "page_end": int,
                    "score": float,
                    ...
                }
            }

    Returns:
        Formatted context string with source metadata and content.
        Returns empty string if no results with content are found.

    Example output:
        Source: ID: doc123, Title: User Manual, Section: Introduction, Page 5
        Content:
        This is the document content...

        Source: ID: doc456, Title: Guide, Pages 10-12
        Content:
        More content here...
    """
    context_parts = []

    for result in search_results:
        content = result.get("content", "")
        metadata = result.get("metadata", {})

        if content:
            # Format page reference
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            if page_start is not None and page_end is not None:
                if page_start == page_end:
                    page_ref = f"Page {page_start}"
                else:
                    page_ref = f"Pages {page_start}-{page_end}"
            else:
                page_ref = "Page N/A"

            # Build source metadata info
            source_info = []
            if metadata.get("doc_id"):
                source_info.append(f"ID: {metadata['doc_id']}")
            if metadata.get("doc_title"):
                source_info.append(f"Title: {metadata['doc_title']}")
            if metadata.get("section_title"):
                source_info.append(f"Section: {metadata['section_title']}")
            if metadata.get("section_path"):
                source_info.append(f"Path: {metadata['section_path']}")
            if metadata.get("score"):
                source_info.append(f"Score: {metadata['score']}")

            # Build final formatted block
            joined_sources = '\n'.join(source_info)
            context_parts.append(
                f"Source: {joined_sources}, {page_ref}\nContent:\n{content}"
            )

    return "\n\n".join(context_parts)
