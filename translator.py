# translator.py
"""
Safe hybrid translator (single-file)
- Produces 5 method-specific translations + 1 hybrid
- Uses local semantic search only (Chroma collections)
- No LLM calls, no hallucination
- Idioms allowed by meaning but require high similarity
- Cites quotes/contexts from datasets when available
"""

import re
from typing import List, Dict, Any, Tuple, Optional


def _safe_similarity_search_with_score(collection, query: str, k: int = 4) -> List[Tuple[Any, float]]:
    """
    Helper: try to call similarity_search_with_score if available,
    otherwise fall back to similarity_search and assign pseudo-scores.
    Returns list of (doc, score) sorted descending by score.
    """
    try:
        results = collection.similarity_search_with_score(query, k=k)
        # expected format: list of (doc, score)
        return results
    except Exception:
        try:
            docs = collection.similarity_search(query, k=k)
            # fallback: assign descending heuristic scores
            out = []
            initial = 0.6
            step = 0.08
            for i, d in enumerate(docs):
                score = initial - i * step
                out.append((d, max(score, 0.0)))
            return out
        except Exception:
            return []


class Translator:
    def __init__(self, exact_dicts: Dict[str, Dict[str, str]], chroma_collections: Dict[str, Any],
                 sound_changes_list: List[Dict] = None, vocabulary_list: List[Dict] = None,
                 grammar_list: List[Dict] = None, fixed_phrases_list: List[Dict] = None):
        """
        exact_dicts: dict loaded from exact_dicts.json, containing keys:
            - vocabulary_modern_to_ancient
            - sound_change_modern_to_ancient
            - fixed_phrases_modern_to_ancient
            ... (and reverse maps)
        chroma_collections: dict of Chroma collections { 'vocab', 'sound_change', 'grammar_pattern', 'fixed_phrase' }
        lists: optional lists loaded from the processed JSON (sound_changes.json, vocabulary.json, etc.)
        """
        self.exact = exact_dicts or {}
        self.chroma = chroma_collections or {}

        # Lowercase keys for robust matching
        self._normalize_exact_keys()

        # Build simple phonology rules from sound_changes_list if provided
        self.sound_changes_map = {}
        if sound_changes_list:
            for sc in sound_changes_list:
                # sc expected to have 'modern' and 'ancient'
                modern = str(sc.get("modern", "")).strip().lower()
                ancient = str(sc.get("ancient", "")).strip()
                if modern:
                    self.sound_changes_map[modern] = ancient

        # Keep lists for citations if provided
        self.sound_changes_list = sound_changes_list or []
        self.vocabulary_list = vocabulary_list or []
        self.grammar_list = grammar_list or []
        self.fixed_phrases_list = fixed_phrases_list or []

        # thresholds (tunable)
        # accept vocab semantic matches above this
        self.vocab_semantic_threshold = 0.35
        self.idiom_semantic_threshold = 0.70      # idioms require high similarity
        self.max_ngram = 3                        # check up to trigrams

    def _normalize_exact_keys(self):
        # Lowercase keys for each mapping inside exact
        for k, v in list(self.exact.items()):
            new = {}
            for mk, mv in v.items():
                new[mk.strip().lower()] = mv
            self.exact[k] = new

    # -------------------------
    # Tokenization & n-grams
    # -------------------------
    def tokenize_with_ngrams(self, text: str) -> List[str]:
        # Simple whitespace tokenization plus punctuation, preserve order.
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        normalized_words = words  # keep punctuation tokens too
        ngrams = []
        L = len(normalized_words)
        for i in range(L):
            # single token
            ngrams.append(normalized_words[i])
            # bigram, trigram (without crossing punctuation ideally)
            for n in range(2, self.max_ngram + 1):
                if i + n <= L:
                    chunk = normalized_words[i:i+n]
                    # skip chunks that include punctuation-only tokens
                    if all(not re.match(r'^[^\w\s]$', c) for c in chunk):
                        ngrams.append(" ".join(chunk))
        # Deduplicate preserving first occurrence
        seen = set()
        out = []
        for g in ngrams:
            key = g.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(g)
        return out

    # -------------------------
    # 1) Exact dictionary match
    # -------------------------
    def exact_based_translation(self, text: str) -> Dict:
        """
        If any modern substring (word or phrase) exactly matches keys in exact dicts,
        replace those substrings with the exact ancient equivalents.
        Returns candidate translation and citations.
        """
        tokens = self.tokenize_with_ngrams(text)
        working = text
        citations = []
        found_any = False

        # Replace longer ngrams first (to avoid partial overlaps)
        tokens_sorted = sorted(tokens, key=lambda s: -len(s.split()))
        for tok in tokens_sorted:
            key = tok.strip().lower()
            # check fixed phrases first (exact meaning match)
            fp_map = self.exact.get("fixed_phrases_modern_to_ancient", {})
            if key in fp_map:
                ancient = fp_map[key]
                # replace whole-word occurrences (case-insensitive)
                working = re.sub(r'\b' + re.escape(tok) + r'\b',
                                 ancient, working, flags=re.IGNORECASE)
                citations.append({"source": "fixed_phrases",
                                 "modern": tok, "ancient": ancient})
                found_any = True
            # vocabulary exact
            vocab_map = self.exact.get("vocabulary_modern_to_ancient", {})
            if key in vocab_map:
                ancient = vocab_map[key]
                working = re.sub(r'\b' + re.escape(tok) + r'\b',
                                 ancient, working, flags=re.IGNORECASE)
                citations.append(
                    {"source": "vocabulary", "modern": tok, "ancient": ancient})
                found_any = True
            # sound change exact (word-level)
            sc_map = self.exact.get("sound_change_modern_to_ancient", {})
            if key in sc_map:
                ancient = sc_map[key]
                working = re.sub(r'\b' + re.escape(tok) + r'\b',
                                 ancient, working, flags=re.IGNORECASE)
                citations.append(
                    {"source": "sound_change", "modern": tok, "ancient": ancient})
                found_any = True

        return {
            "translation": working,
            "confidence": 1.0 if found_any else 0.0,
            "citations": citations,
            "method": "exact"
        }

    # -------------------------
    # 2) Phonology / Sound-change
    # -------------------------
    def phonology_based_translation(self, text: str) -> Dict:
        """
        Apply two phonology strategies:
        - exact sound_change mapping for whole words (from exact_dicts or sound_changes_list)
        - fallback: rule-based regex replacements from sound_changes_list (if available)
        """
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        out_tokens = []
        citations = []

        for token in tokens:
            if re.match(r'^[^\w\s]$', token):  # punctuation
                out_tokens.append(token)
                continue

            lower = token.lower()
            mapped = None

            # 1) exact sound change mapping (exact_dict)
            sc_map = self.exact.get("sound_change_modern_to_ancient", {})
            if lower in sc_map:
                mapped = sc_map[lower]
                citations.append(
                    {"source": "sound_change_exact", "modern": token, "ancient": mapped})
                out_tokens.append(mapped)
                continue

            # 2) look in provided sound_changes_list heuristic map
            if lower in self.sound_changes_map:
                mapped = self.sound_changes_map[lower]
                citations.append(
                    {"source": "sound_change_list", "modern": token, "ancient": mapped})
                out_tokens.append(mapped)
                continue

            # 3) regex-style phonology rules (best-effort, low confidence)
            transformed = self._apply_phonology_regex(token)
            if transformed != token:
                citations.append({"source": "phonology_rules",
                                 "modern": token, "ancient": transformed})
                out_tokens.append(transformed)
                continue

            # fallback: keep
            out_tokens.append(token)

        return {
            "translation": " ".join(out_tokens),
            "confidence": 0.7,
            "citations": citations,
            "method": "phonology"
        }

    def _apply_phonology_regex(self, w: str) -> str:
        # Best-effort replacements — conservative and reversible
        s = w
        # Example simple reversible rules (safe, conservative)
        # These are intentionally few — you can extend with more precise rules
        rules = [
            (r'iê\b', 'ê'),   # iê -> ê (reverse of diphthongization)
            (r'uyê\b', 'uê'),
            (r'uô\b', 'ô'),
            (r'ươ\b', 'ơ'),
            (r'\bay\b', 'ăy'),
        ]
        for pat, rep in rules:
            s2 = re.sub(pat, rep, s, flags=re.IGNORECASE)
            if s2 != s:
                return s2
        return s

    # -------------------------
    # 3) Vocabulary-based (synonyms via local semantic search)
    # -------------------------
    def vocabulary_based_translation(self, text: str) -> Dict:
        """
        For each token/ngram, search the vocab collection (semantic search)
        to find a modern -> ancient mapping. Use a threshold to avoid bad matches.
        """
        tokens = self.tokenize_with_ngrams(text)
        working = text
        citations = []
        found = False

        # attempt longer ngrams first
        tokens_sorted = sorted(tokens, key=lambda s: -len(s.split()))
        for tok in tokens_sorted:
            key = tok.strip()
            if not key or re.match(r'^[^\w\s]+$', key):
                continue

            # check exact vocabulary mapping first
            vm = self.exact.get("vocabulary_modern_to_ancient", {})
            if key.lower() in vm:
                ancient = vm[key.lower()]
                working = re.sub(r'\b' + re.escape(key) + r'\b',
                                 ancient, working, flags=re.IGNORECASE)
                citations.append(
                    {"source": "vocab_exact", "modern": key, "ancient": ancient})
                found = True
                continue

            # semantic search inside vocab collection
            vocab_coll = self.chroma.get("vocab")
            if vocab_coll:
                results = _safe_similarity_search_with_score(
                    vocab_coll, key, k=5)
                if results:
                    best_doc, best_score = results[0]
                    if best_score >= self.vocab_semantic_threshold:
                        # get ancient form from doc.metadata or doc.page_content
                        ancient_candidate = None
                        meta = getattr(best_doc, "metadata", {}) or {}
                        page_content = getattr(best_doc, "page_content", None)
                        # prefer metadata.ancient if exists, else page_content
                        if meta.get("ancient"):
                            ancient_candidate = meta.get("ancient")
                        elif page_content:
                            ancient_candidate = page_content
                        else:
                            ancient_candidate = vm.get(best_doc, None)

                        if ancient_candidate:
                            working = re.sub(
                                r'\b' + re.escape(key) + r'\b', ancient_candidate, working, flags=re.IGNORECASE)
                            citations.append(
                                {"source": "vocab_semantic", "modern": key, "ancient": ancient_candidate, "score": best_score})
                            found = True
                            continue

        return {
            "translation": working,
            "confidence": 0.85 if found else 0.0,
            "citations": citations,
            "method": "vocabulary_semantic"
        }

    # -------------------------
    # 4) Grammar-based translation (pattern replacements)
    # -------------------------
    def grammar_based_translation(self, text: str) -> Dict:
        """
        Use grammar_pattern collection to find sentence-level pattern matches
        and apply replacements. Conservative: require moderately-high similarity.
        """
        grammar_coll = self.chroma.get("grammar_pattern")
        if not grammar_coll:
            return {"translation": text, "confidence": 0.0, "citations": [], "method": "grammar"}

        results = _safe_similarity_search_with_score(grammar_coll, text, k=4)
        working = text
        citations = []
        applied = False

        for doc, score in results:
            if score < 0.35:
                continue
            meta = getattr(doc, "metadata", {}) or {}
            modern_pat = meta.get("modern", meta.get("modern_form", ""))
            ancient_pat = meta.get("ancient", meta.get("ancient_form", ""))
            if modern_pat and ancient_pat:
                # replace occurrences of modern pattern conservatively
                working_new = working.replace(modern_pat, ancient_pat)
                if working_new != working:
                    working = working_new
                    citations.append(
                        {"source": "grammar", "modern": modern_pat, "ancient": ancient_pat, "score": score})
                    applied = True

        return {
            "translation": working,
            "confidence": 0.6 if applied else 0.0,
            "citations": citations,
            "method": "grammar"
        }

    # -------------------------
    # 5) Idiom-based (fixed phrases) — allow meaning-match but strict threshold
    # -------------------------
    def idiom_based_translation(self, text: str) -> Dict:
        """
        Try to map idioms (fixed phrases). We allow:
         - exact modern string matches (safe)
         - semantic matches via fixed_phrase collection BUT require HIGH threshold
        """
        working = text
        citations = []
        found = False

        # 1) exact modern -> ancient via exact_dict
        fp_map = self.exact.get("fixed_phrases_modern_to_ancient", {})
        # check longer phrases first
        tokens = sorted(self.tokenize_with_ngrams(
            text), key=lambda s: -len(s.split()))
        for tok in tokens:
            key = tok.strip().lower()
            if key in fp_map:
                ancient = fp_map[key]
                working = re.sub(r'\b' + re.escape(tok) + r'\b',
                                 ancient, working, flags=re.IGNORECASE)
                citations.append(
                    {"source": "fixed_exact", "modern": tok, "ancient": ancient})
                found = True

        if found:
            return {"translation": working, "confidence": 0.98, "citations": citations, "method": "idiom_exact"}

        # 2) semantic search on fixed_phrase collection
        fixed_coll = self.chroma.get("fixed_phrase")
        if fixed_coll:
            results = _safe_similarity_search_with_score(fixed_coll, text, k=5)
            if results:
                best_doc, best_score = results[0]
                if best_score >= self.idiom_semantic_threshold:
                    meta = getattr(best_doc, "metadata", {}) or {}
                    ancient = meta.get("ancient") or getattr(
                        best_doc, "page_content", None)
                    if ancient:
                        # apply only if it's safe (score high)
                        working = ancient
                        citations.append(
                            {"source": "fixed_semantic", "modern_query": text, "ancient": ancient, "score": best_score})
                        return {"translation": working, "confidence": best_score, "citations": citations, "method": "idiom_semantic"}

        return {"translation": working, "confidence": 0.0, "citations": [], "method": "idiom_none"}

    # -------------------------
    # Hybrid combination: per-word best candidate
    # -------------------------
    def hybrid_translation(self, text: str) -> Dict:
        """
        Build per-word candidates from exact, vocabulary, sound_change, phonology.
        For each word/phrase choose the best candidate according to a priority
        and assemble final text. Also add a combined citations list.
        """
        # produce candidate maps per token (use base tokenization)
        base_tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        final_tokens = []
        all_citations = []
        confidences = []

        for token in base_tokens:
            if re.match(r'^[^\w\s]$', token):  # punctuation
                final_tokens.append(token)
                continue

            lowered = token.lower()

            # Candidate 1: exact fixed phrase (if token is multiword this won't match single token)
            exact_v = self.exact.get(
                "fixed_phrases_modern_to_ancient", {}).get(lowered)
            if exact_v:
                final_tokens.append(exact_v)
                all_citations.append(
                    {"source": "exact_fixed", "modern": token, "ancient": exact_v})
                confidences.append(1.0)
                continue

            # Candidate 2: exact vocabulary
            exact_vocab = self.exact.get(
                "vocabulary_modern_to_ancient", {}).get(lowered)
            if exact_vocab:
                final_tokens.append(exact_vocab)
                all_citations.append(
                    {"source": "exact_vocab", "modern": token, "ancient": exact_vocab})
                confidences.append(0.95)
                continue

            # Candidate 3: exact sound change
            exact_sc = self.exact.get(
                "sound_change_modern_to_ancient", {}).get(lowered)
            if exact_sc:
                final_tokens.append(exact_sc)
                all_citations.append(
                    {"source": "exact_sound_change", "modern": token, "ancient": exact_sc})
                confidences.append(0.9)
                continue

            # Candidate 4: semantic vocab search
            best_vocab_candidate, score = self._semantic_vocab_lookup(token)
            if best_vocab_candidate and score >= self.vocab_semantic_threshold:
                final_tokens.append(best_vocab_candidate)
                all_citations.append(
                    {"source": "semantic_vocab", "modern": token, "ancient": best_vocab_candidate, "score": score})
                confidences.append(0.85)
                continue

            # Candidate 5: phonology regex / heuristic
            phon = self._apply_phonology_regex(token)
            if phon != token:
                final_tokens.append(phon)
                all_citations.append(
                    {"source": "phonology_rule", "modern": token, "ancient": phon})
                confidences.append(0.65)
                continue

            # Fallback: keep original
            final_tokens.append(token)
            confidences.append(0.3)

        hybrid_text = " ".join(final_tokens)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "translation": hybrid_text,
            "confidence": round(avg_conf, 2),
            "citations": all_citations,
            "method": "hybrid"
        }

    def _semantic_vocab_lookup(self, token: str) -> Tuple[Optional[str], float]:
        """
        Search vocab collection for semantic match for token.
        Return (ancient_candidate, score) if found, else (None, 0).
        """
        vocab_coll = self.chroma.get("vocab")
        if not vocab_coll:
            return None, 0.0

        results = _safe_similarity_search_with_score(vocab_coll, token, k=6)
        if not results:
            return None, 0.0

        best_doc, best_score = results[0]
        meta = getattr(best_doc, "metadata", {}) or {}
        ancient = meta.get("ancient")
        # fallback: page_content might hold the ancient or gloss
        if not ancient:
            ancient = getattr(best_doc, "page_content", None)

        if ancient:
            return ancient, best_score
        return None, 0.0

    # -------------------------
    # Top-level API: produce 5 candidates + hybrid
    # -------------------------
    def translate_all(self, text: str) -> Dict[str, Any]:
        """
        Returns:
        {
          "exact": {...},
          "phonology": {...},
          "vocabulary": {...},
          "grammar": {...},
          "idiom": {...},
          "hybrid": {...},
          "combined_explanation": "..."
        }
        Each entry contains translation, confidence, citations, method
        """
        # 1 exact
        exact_cand = self.exact_based_translation(text)

        # 2 phonology
        phon_cand = self.phonology_based_translation(text)

        # 3 vocabulary semantic
        vocab_cand = self.vocabulary_based_translation(text)

        # 4 grammar
        gram_cand = self.grammar_based_translation(text)

        # 5 idiom
        idiom_cand = self.idiom_based_translation(text)

        # hybrid
        hybrid_cand = self.hybrid_translation(text)

        # combined explanation: list unique citations
        combined_citations = []
        seen = set()
        for cset in [exact_cand, phon_cand, vocab_cand, gram_cand, idiom_cand, hybrid_cand]:
            for cit in cset.get("citations", []):
                key = (cit.get("source"), cit.get(
                    "modern"), cit.get("ancient"))
                if key not in seen:
                    seen.add(key)
                    combined_citations.append(cit)

        explanation = {
            "summary": f"Produced 5 method-specific candidates and 1 hybrid. Hybrid prioritized exact>vocab>sound>phonology.",
            "citations": combined_citations
        }

        return {
            "exact": exact_cand,
            "phonology": phon_cand,
            "vocabulary": vocab_cand,
            "grammar": gram_cand,
            "idiom": idiom_cand,
            "hybrid": hybrid_cand,
            "explanation": explanation
        }
