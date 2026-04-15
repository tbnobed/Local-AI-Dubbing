"""
Translation service using Google MADLAD-400 3B.

MADLAD-400 advantages over NLLB-200:
  - Apache 2.0 license (commercially usable)
  - 400+ languages (vs 200)
  - Better translation quality on benchmarks
  - Uses simple <2xx> prefix tags for target language

Uses T5 architecture. Input format: "<2es> Hello world" → "Hola mundo"
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
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            model_name = self.config.translation_model
            logger.info(f"Loading translation model: {model_name}")

            self._tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir / "madlad"),
            )
            self._model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=str(self.config.models_dir / "madlad"),
                torch_dtype=torch.float16,
            )

            if torch.cuda.is_available() and self.config.use_gpu:
                self._device = torch.device(f"cuda:{self.config.primary_gpu_id}")
            else:
                self._device = torch.device("cpu")

            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"MADLAD-400 loaded on {self._device}")

        return self._model, self._tokenizer

    def translate_text(
        self,
        text: str,
        target_lang: str,
        max_length: int = 512,
    ) -> str:
        """
        Translate text using MADLAD-400.
        MADLAD uses <2xx> prefix: "<2es> Hello" → "Hola"
        """
        import torch

        if not text.strip():
            return text

        model, tokenizer = self._load_model()

        # MADLAD-400 format: prepend target language tag
        madlad_code = self.config.supported_languages.get(target_lang, {}).get("madlad", target_lang)
        input_text = f"<2{madlad_code}> {text}"

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
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
        total = len(segments)

        for i, segment in enumerate(segments):
            translated_text = self.translate_text(
                segment.text,
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
