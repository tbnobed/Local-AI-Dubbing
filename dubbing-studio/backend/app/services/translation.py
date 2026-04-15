"""
Translation service using Meta NLLB-200 3.3B.

NLLB-200 (No Language Left Behind) uses BCP-47 style language codes
with script suffixes, e.g. "spa_Latn" for Spanish, "fra_Latn" for French.
The target language is set via the tokenizer's forced BOS token.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "hi": "hin_Deva",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ro": "ron_Latn",
    "sk": "slk_Latn",
    "uk": "ukr_Cyrl",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "hr": "hrv_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "fa": "pes_Arab",
    "sw": "swh_Latn",
    "tl": "tgl_Latn",
}


class TranslationService:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = self.config.translation_model
            logger.info(f"Loading translation model: {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir / "nllb"),
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir / "nllb"),
                torch_dtype=torch.float16,
            )

            if torch.cuda.is_available() and self.config.use_gpu:
                if torch.cuda.device_count() > 1:
                    self._device = torch.device(f"cuda:{self.config.secondary_gpu_id}")
                else:
                    self._device = torch.device(f"cuda:{self.config.primary_gpu_id}")
            else:
                self._device = torch.device("cpu")

            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"NLLB-200 loaded on {self._device}")

        return self._model, self._tokenizer

    def _get_nllb_code(self, lang: str) -> str:
        if lang in NLLB_LANG_CODES:
            return NLLB_LANG_CODES[lang]
        nllb_from_config = self.config.supported_languages.get(lang, {}).get("nllb")
        if nllb_from_config:
            return nllb_from_config
        return lang

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
    ) -> str:
        import torch

        if not text.strip():
            return text

        model, tokenizer = self._load_model()

        src_code = self._get_nllb_code(source_lang)
        tgt_code = self._get_nllb_code(target_lang)

        tokenizer.src_lang = src_code

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return result

    def translate_segments(
        self,
        segments: list,
        source_lang: str,
        target_lang: str,
        progress_callback=None,
    ) -> list:
        self._load_model()

        total = len(segments)

        for i, segment in enumerate(segments):
            translated_text = self.translate_text(
                segment.text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            segment.translated_text = translated_text

            if progress_callback:
                progress_callback((i + 1) / total)

        logger.info(f"Translated {total} segments from {source_lang} to {target_lang}")
        return segments

    def unload(self):
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Translation model unloaded.")
