import re
from typing import List, Dict, Any, Optional


class DocumentResolver:
    """
    Parses user queries to detect target document specifications for document-scoped summarization.
    Resolves candidates against known indexed documents in the knowledge base, detecting ambiguity.
    """

    KNOWN_EXTENSIONS = ('.docx', '.pdf', '.txt', '.xlsx', '.pptx', '.csv')

    def __init__(self, known_documents: Optional[List[str]] = None):
        self.known_documents = known_documents or []

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """
        Normalizes a document filename or query term for resilient matching:
        - Removes file extensions
        - Removes leading index prefixes like '1_', '02.', '4_'
        - Replaces underscores and hyphens with spaces
        - Collapses whitespace and lowercases
        """
        s = name.strip().lower()
        s = re.sub(r'\.[a-zA-Z0-9]+$', '', s)
        s = re.sub(r'^\d+[\._\s]*', '', s)
        s = re.sub(r'[\-_\.]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def resolve(self, query: str, candidate_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Resolves query to a target document.
        Returns:
            {"status": "resolved", "target_document": "<filename>"}
            {"status": "ambiguous", "matches": [...], "query_term": "...", "clarification_message": "..."}
            {"status": "not_found", "target_document": "<filename>"}
            {"status": "none"}
        """
        docs = candidate_docs if candidate_docs is not None else self.known_documents
        if not query or not query.strip():
            return {"status": "none"}

        q_lower = query.strip().lower()

        # -------------------------------------------------------------
        # Pass 1: Explicit file extension pattern in query (e.g. SRS_part1.docx)
        # -------------------------------------------------------------
        file_ext_match = re.search(r'\b([\w\-\.]+\.(?:docx|pdf|txt|xlsx|pptx))\b', query, re.IGNORECASE)
        if file_ext_match:
            raw_target = file_ext_match.group(1)
            target_norm = raw_target.lower()
            target_no_pfx = re.sub(r'^\d+_', '', target_norm)

            matched = []
            for d in docs:
                d_norm = d.lower()
                d_no_pfx = re.sub(r'^\d+_', '', d_norm)
                if target_norm == d_norm or target_norm == d_no_pfx or target_no_pfx == d_norm or target_no_pfx == d_no_pfx:
                    matched.append(d)

            if len(matched) == 1:
                return {"status": "resolved", "target_document": matched[0]}
            elif len(matched) > 1:
                matches = sorted(list(set(matched)))
                return {
                    "status": "ambiguous",
                    "matches": matches,
                    "query_term": raw_target,
                    "clarification_message": self._format_clarification(raw_target, matches)
                }
            else:
                return {"status": "not_found", "target_document": raw_target}

        # -------------------------------------------------------------
        # Pass 2: Exact matching of full known filenames inside query
        # -------------------------------------------------------------
        exact_matches = []
        for d in docs:
            d_clean = d.lower()
            d_no_pfx = re.sub(r'^\d+_', '', d_clean)
            for candidate in [d_clean, d_no_pfx]:
                pattern = r'(?<![\w\.\-])' + re.escape(candidate) + r'(?![\w\.\-])'
                if re.search(pattern, q_lower):
                    exact_matches.append(d)
                    break

        if len(exact_matches) == 1:
            return {"status": "resolved", "target_document": exact_matches[0]}
        elif len(exact_matches) > 1:
            matches = sorted(list(set(exact_matches)))
            return {
                "status": "ambiguous",
                "matches": matches,
                "query_term": "multiple documents",
                "clarification_message": self._format_clarification("multiple documents", matches)
            }

        # -------------------------------------------------------------
        # Pass 3: Stem / Title / Substring Matching (without extension)
        # e.g., "SRS_part1", "CRM SRS part 1", "Product Roadmap"
        # -------------------------------------------------------------
        stem_matches = []
        matched_term = None

        for d in docs:
            raw_stem = re.sub(r'\.[a-zA-Z0-9]+$', '', d)
            stem_norm = self.normalize_name(d)
            stem_lower = raw_stem.lower()
            stem_no_pfx = re.sub(r'^\d+_', '', stem_lower)

            # Check raw stem bounded: e.g. "srs_part1"
            patterns = [
                r'(?<![\w\.\-])' + re.escape(stem_lower) + r'(?![\w\.\-])',
                r'(?<![\w\.\-])' + re.escape(stem_no_pfx) + r'(?![\w\.\-])',
            ]
            if any(re.search(p, q_lower) for p in patterns):
                stem_matches.append(d)
                matched_term = raw_stem
                continue

            # Check normalized space-separated stem (e.g. "srs part1" or "product roadmap")
            if len(stem_norm) >= 4 and stem_norm not in {'data', 'user', 'flow', 'part', 'test', 'guide', 'module'}:
                if re.search(r'\b' + re.escape(stem_norm) + r'\b', q_lower):
                    stem_matches.append(d)
                    matched_term = raw_stem
                    continue
                if stem_norm.replace(" ", "") in q_lower.replace(" ", ""):
                    stem_matches.append(d)
                    matched_term = raw_stem
                    continue

        stem_matches = sorted(list(set(stem_matches)))
        if len(stem_matches) == 1:
            return {"status": "resolved", "target_document": stem_matches[0]}
        elif len(stem_matches) > 1:
            return {
                "status": "ambiguous",
                "matches": stem_matches,
                "query_term": matched_term or "document",
                "clarification_message": self._format_clarification(matched_term or "the document name", stem_matches)
            }

        # -------------------------------------------------------------
        # Pass 4: Prefix / Substring match for potential partial names
        # e.g., "Summarize SRS_part1" when candidates are ['SRS_part1_v1.docx', 'SRS_part1_v2.docx']
        # -------------------------------------------------------------
        candidate_words = [w for w in re.split(r'[\s,;:()]+', q_lower) if len(w) >= 4 and w not in {'summarize', 'content', 'including', 'definitions', 'references', 'about', 'document', 'module', 'system'}]
        for word in candidate_words:
            partial_matches = []
            for d in docs:
                d_stem = re.sub(r'\.[a-zA-Z0-9]+$', '', d).lower()
                d_no_pfx = re.sub(r'^\d+_', '', d_stem)
                if word in d_stem or word in d_no_pfx:
                    partial_matches.append(d)
            partial_matches = sorted(list(set(partial_matches)))
            if len(partial_matches) > 1:
                return {
                    "status": "ambiguous",
                    "matches": partial_matches,
                    "query_term": word,
                    "clarification_message": self._format_clarification(word, partial_matches)
                }
            elif len(partial_matches) == 1:
                return {"status": "resolved", "target_document": partial_matches[0]}

        # -------------------------------------------------------------
        # Pass 5: Generic Acronym / Partial checks (e.g. "SRS" matching all SRS docs)
        # -------------------------------------------------------------
        if re.search(r'\bsrs\b', q_lower):
            srs_matches = sorted([d for d in docs if 'srs' in d.lower()])
            if len(srs_matches) > 1:
                return {
                    "status": "ambiguous",
                    "matches": srs_matches,
                    "query_term": "SRS",
                    "clarification_message": self._format_clarification("SRS", srs_matches)
                }
            elif len(srs_matches) == 1:
                return {"status": "resolved", "target_document": srs_matches[0]}

        return {"status": "none"}

    def _format_clarification(self, term: str, matches: List[str]) -> str:
        """Formats an unambiguous, numbered clarification request."""
        options = "\n".join([f"{i+1}. {m}" for i, m in enumerate(matches)])
        return (
            f'I found multiple documents matching "{term}".\n'
            f'Please select the document you want me to summarize:\n'
            f'{options}'
        )
