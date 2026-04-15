"""
Translation service using NLLB-200 (No Language Left Behind) by Meta.
Supports 200+ languages, runs fully locally on GPU.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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

            logger.info(f"Loading translation model: {self.config.translation_model}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.translation_model,
                cache_dir=str(self.config.models_dir / "nllb"),
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.translation_model,
                cache_dir=str(self.config.models_dir / "nllb"),
            )

            if torch.cuda.is_available() and self.config.use_gpu:
                self._device = torch.device(f"cuda:{self.config.primary_gpu_id}")
            else:
                self._device = torch.device("cpu")

            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Translation model loaded on {self._device}")

        return self._model, self._tokenizer

    def translate_text(
        self,
        text: str,
        source_lang_nllb: str,
        target_lang_nllb: str,
        max_length: int = 512,
    ) -> str:
        import torch

        if not text.strip():
            return text

        model, tokenizer = self._load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_nllb],
                max_length=max_length,
                num_beams=4,
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
        source_nllb = self.config.supported_languages[source_lang]["nllb"]
        target_nllb = self.config.supported_languages[target_lang]["nllb"]

        total = len(segments)
        translated = []

        for i, segment in enumerate(segments):
            translated_text = self.translate_text(
                segment.text,
                source_lang_nllb=source_nllb,
                target_lang_nllb=target_nllb,
            )
            segment.translated_text = translated_text
            translated.append(segment)

            if progress_callback:
                progress_callback((i + 1) / total)

        logger.info(f"Translated {total} segments from {source_lang} to {target_lang}")
        return translated

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
