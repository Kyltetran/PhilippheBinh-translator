import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI


@dataclass
class TranslationCandidate:
    """Lưu trữ ứng viên dịch thuật"""
    original: str
    translated: str
    method: str  # 'exact', 'sound', 'vocab', 'grammar', 'idiom'
    source: str
    quote: str
    confidence: float


class PhilippeBinhTranslator:
    def __init__(self, data_dir: str = "data", chroma_path: str = "./chroma_db"):
        self.data_dir = data_dir
        self.chroma_path = chroma_path
        self.client = Chroma.PersistentClient(path=chroma_path)
        self.openai_client = ChatOpenAI()

        # Load all data
        self.exact_dicts = self._load_json("exact_dicts.json")
        self.sound_changes = self._load_json("sound_changes.json")
        self.vocabulary = self._load_json("vocabulary.json")
        self.grammar_patterns = self._load_json("grammar_patterns.json")
        self.fixed_phrases = self._load_json("fixed_phrases.json")

        # Get collections (handle missing collections gracefully)
        try:
            self.vocab_collection = self.client.get_collection("vocabulary")
        except:
            print("⚠️  Warning: vocabulary collection not found")
            self.vocab_collection = None

        try:
            self.grammar_collection = self.client.get_collection(
                "grammar_patterns")
        except:
            print("⚠️  Warning: grammar_patterns collection not found")
            self.grammar_collection = None

        try:
            self.phrase_collection = self.client.get_collection(
                "fixed_phrases")
        except:
            print("⚠️  Warning: fixed_phrases collection not found")
            self.phrase_collection = None

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file từ data directory"""
        path = os.path.join(self.data_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_llm_synonyms(self, word: str, context: str = "") -> List[str]:
        """Dùng LLM tìm từ đồng nghĩa/gần nghĩa trong tiếng Việt"""
        prompt = f"""Cho từ/cụm từ tiếng Việt: "{word}"
{f'Trong ngữ cảnh: "{context}"' if context else ''}

Hãy liệt kê các từ đồng nghĩa, gần nghĩa, hoặc cách diễn đạt tương tự trong tiếng Việt.
Chỉ trả về danh sách các từ, mỗi từ một dòng, không giải thích.
Tối đa 10 từ."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        synonyms = response.choices[0].message.content.strip().split('\n')
        return [s.strip('- ').strip() for s in synonyms if s.strip()]

    def translate_word_exact(self, word: str) -> Optional[TranslationCandidate]:
        """Phương pháp 1: Exact match từ exact_dicts"""
        # Sound change exact
        if word in self.exact_dicts.get("sound_change_modern_to_ancient", {}):
            ancient = self.exact_dicts["sound_change_modern_to_ancient"][word]
            return TranslationCandidate(
                original=word,
                translated=ancient,
                method="exact_sound",
                source="exact_dicts.json",
                quote="",
                confidence=1.0
            )

        # Vocabulary exact
        if word in self.exact_dicts.get("vocabulary_modern_to_ancient", {}):
            ancient = self.exact_dicts["vocabulary_modern_to_ancient"][word]
            return TranslationCandidate(
                original=word,
                translated=ancient,
                method="exact_vocab",
                source="exact_dicts.json",
                quote="",
                confidence=1.0
            )

        return None

    def translate_word_sound(self, word: str) -> List[TranslationCandidate]:
        """Phương pháp 2: Sound change rules
        Không dùng synonym vì đây là quy tắc âm học
        """
        candidates = []

        for item in self.sound_changes:
            modern = item.get("modern", "")
            ancient = item.get("ancient", "")
            rule = item.get("rule", "")

            # Kiểm tra nếu word chứa âm modern
            if modern in word:
                # Thay thế âm
                translated = word.replace(modern, ancient)
                candidates.append(TranslationCandidate(
                    original=word,
                    translated=translated,
                    method="sound_rule",
                    source=f"Rule: {rule}",
                    quote=item.get("quote", ""),
                    confidence=0.7  # Lower confidence vì có thể không chính xác 100%
                ))

        return candidates

    def translate_word_vocab(self, word: str, sentence: str = "") -> List[TranslationCandidate]:
        """Phương pháp 3: Vocabulary với synonym search"""
        candidates = []

        # 1. Tìm trực tiếp trong vocabulary.json
        for item in self.vocabulary:
            if word in item.get("modern", []):
                candidates.append(TranslationCandidate(
                    original=word,
                    translated=item["ancient"],
                    method="vocab_direct",
                    source="vocabulary.json",
                    quote=item.get("quote", [{}])[0].get("context", ""),
                    confidence=0.9
                ))

        # 2. Nếu không tìm thấy, dùng LLM tìm synonym
        if not candidates:
            synonyms = self._get_llm_synonyms(word, sentence)

            # Search trong ChromaDB với synonyms
            for syn in synonyms:
                results = self.vocab_collection.query(
                    query_texts=[syn],
                    n_results=3
                )

                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        distance = results['distances'][0][i]

                        candidates.append(TranslationCandidate(
                            original=word,
                            translated=metadata.get('ancient', ''),
                            method="vocab_synonym",
                            source=f"Via synonym: {syn}",
                            quote=metadata.get('quote', ''),
                            # Giảm confidence theo khoảng cách
                            confidence=max(0.3, 0.9 - distance)
                        ))

        return candidates

    def translate_phrase_grammar(self, sentence: str) -> List[TranslationCandidate]:
        """Phương pháp 4: Grammar patterns"""
        candidates = []

        # 1. Tìm trực tiếp
        for item in self.grammar_patterns:
            modern = item.get("modern", "")
            if modern in sentence:
                candidates.append(TranslationCandidate(
                    original=modern,
                    translated=item["ancient"],
                    method="grammar_direct",
                    source="grammar_patterns.json",
                    quote=item.get("quote", [{}])[0].get("context", ""),
                    confidence=0.9
                ))

        # 2. Tìm với synonym
        if not candidates:
            # Tách sentence thành các cụm từ ngữ pháp có thể
            phrases = self._extract_grammar_phrases(sentence)

            for phrase in phrases:
                synonyms = self._get_llm_synonyms(phrase, sentence)

                for syn in synonyms:
                    results = self.grammar_collection.query(
                        query_texts=[syn],
                        n_results=2
                    )

                    if results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results['metadatas'][0][i]
                            distance = results['distances'][0][i]

                            candidates.append(TranslationCandidate(
                                original=phrase,
                                translated=metadata.get('ancient', ''),
                                method="grammar_synonym",
                                source=f"Via synonym: {syn}",
                                quote=metadata.get('quote', ''),
                                confidence=max(0.3, 0.85 - distance)
                            ))

        return candidates

    def translate_phrase_idiom(self, sentence: str) -> List[TranslationCandidate]:
        """Phương pháp 5: Fixed phrases/idioms"""
        candidates = []

        # 1. Tìm trực tiếp
        for item in self.fixed_phrases:
            modern = item.get("modern", "")
            if modern.lower() in sentence.lower():
                candidates.append(TranslationCandidate(
                    original=modern,
                    translated=item["ancient"],
                    method="idiom_direct",
                    source="fixed_phrases.json",
                    quote=item.get("quote", [{}])[0].get("context", ""),
                    confidence=0.95
                ))

        # 2. Tìm với LLM semantic matching
        if not candidates:
            # Dùng LLM tìm idiom tương tự
            idiom_candidates = self._find_similar_idioms(sentence)
            candidates.extend(idiom_candidates)

        return candidates

    def _find_similar_idioms(self, sentence: str) -> List[TranslationCandidate]:
        """Dùng LLM tìm thành ngữ tương tự"""
        candidates = []

        # Lấy danh sách idioms
        idiom_list = "\n".join(
            [f"- {item['modern']}" for item in self.fixed_phrases[:20]])

        prompt = f"""Câu: "{sentence}"

Danh sách thành ngữ có sẵn:
{idiom_list}

Có thành ngữ nào trong danh sách có ý nghĩa giống hoặc liên quan đến câu trên không?
Nếu có, trả về tên thành ngữ đó. Nếu không, trả về "NONE"."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        result = response.choices[0].message.content.strip()

        if result != "NONE":
            # Tìm idiom trong database
            for item in self.fixed_phrases:
                if item["modern"] in result:
                    candidates.append(TranslationCandidate(
                        original=sentence,
                        translated=item["ancient"],
                        method="idiom_semantic",
                        source=f"Semantic match: {item['modern']}",
                        quote=item.get("quote", [{}])[0].get("context", ""),
                        confidence=0.6
                    ))

        return candidates

    def _extract_grammar_phrases(self, sentence: str) -> List[str]:
        """Trích xuất các cụm từ ngữ pháp có thể từ câu"""
        # Simple heuristic: n-grams
        words = sentence.split()
        phrases = []

        for n in range(2, min(6, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                phrases.append(phrase)

        return phrases

    def translate_sentence(self, sentence: str) -> Dict:
        """Dịch cả câu với tất cả phương pháp"""
        results = {
            "original": sentence,
            "translations": {
                "exact": [],
                "sound": [],
                "vocab": [],
                "grammar": [],
                "idiom": [],
                "combined": ""
            }
        }

        words = sentence.split()

        # 1. Exact matches cho từng từ
        for word in words:
            exact = self.translate_word_exact(word)
            if exact:
                results["translations"]["exact"].append(exact)

        # 2. Sound changes
        for word in words:
            sounds = self.translate_word_sound(word)
            results["translations"]["sound"].extend(sounds)

        # 3. Vocabulary
        for word in words:
            vocabs = self.translate_word_vocab(word, sentence)
            results["translations"]["vocab"].extend(vocabs)

        # 4. Grammar patterns
        grammars = self.translate_phrase_grammar(sentence)
        results["translations"]["grammar"].extend(grammars)

        # 5. Idioms
        idioms = self.translate_phrase_idiom(sentence)
        results["translations"]["idiom"].extend(idioms)

        # 6. Combined translation
        results["translations"]["combined"] = self._create_combined_translation(
            sentence, results["translations"]
        )

        return results

    def _create_combined_translation(self, sentence: str, translations: Dict) -> str:
        """Kết hợp tất cả phương pháp để tạo bản dịch tối ưu"""
        # Sort by confidence
        all_candidates = []
        for method, candidates in translations.items():
            if method != "combined":
                all_candidates.extend(candidates)

        # Group by original word/phrase
        grouped = {}
        for cand in all_candidates:
            if cand.original not in grouped:
                grouped[cand.original] = []
            grouped[cand.original].append(cand)

        # Pick best candidate for each
        best_translations = {}
        for original, candidates in grouped.items():
            best = max(candidates, key=lambda x: x.confidence)
            best_translations[original] = best.translated

        # Replace in sentence (longest first to avoid partial matches)
        result = sentence
        for original in sorted(best_translations.keys(), key=len, reverse=True):
            result = result.replace(original, best_translations[original])

        return result

    def format_output(self, results: Dict) -> str:
        """Format kết quả đẹp để hiển thị"""
        output = [
            f"Câu gốc: {results['original']}",
            "",
            "=" * 60,
            ""
        ]

        methods = {
            "exact": "1. EXACT MATCH",
            "sound": "2. SOUND CHANGES",
            "vocab": "3. VOCABULARY",
            "grammar": "4. GRAMMAR PATTERNS",
            "idiom": "5. IDIOMS/FIXED PHRASES"
        }

        for method, title in methods.items():
            candidates = results["translations"][method]
            if candidates:
                output.append(title)
                output.append("-" * 60)
                for cand in candidates:
                    output.append(f"  {cand.original} → {cand.translated}")
                    output.append(f"  Phương pháp: {cand.method}")
                    output.append(f"  Độ tin cậy: {cand.confidence:.2f}")
                    if cand.quote:
                        output.append(f"  Trích dẫn: {cand.quote[:100]}...")
                    output.append("")

        output.extend([
            "=" * 60,
            "6. BẢN DỊCH KẾT HỢP",
            "-" * 60,
            results["translations"]["combined"],
            ""
        ])

        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    translator = PhilippeBinhTranslator()

    # Test
    sentence = "Tôi đang học tiếng Việt cổ"
    results = translator.translate_sentence(sentence)
    print(translator.format_output(results))
