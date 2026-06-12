from transformers import pipeline, TextClassificationPipeline, AutoTokenizer

try:
    import fasttext
except ImportError:
    fasttext = None


def is_fasttext_available():
    return fasttext is not None


# ─── Pipeline an toàn cho DistilBERT ─────────────────────────────────────────
# DistilBERT không có token_type_ids. Subclass này tự động loại bỏ
# trước khi truyền vào model.forward() để tránh TypeError.
class _DistilBertPipeline(TextClassificationPipeline):
    def _forward(self, model_inputs, **forward_params):
        model_inputs.pop("token_type_ids", None)
        return super()._forward(model_inputs, **forward_params)


class FormalityPredictor:
    def __init__(self, model_type="roberta"):
        self.model_type = model_type

        if self.model_type == "roberta":
            model_path = "./models/best_roberta_formality"
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
            )

        elif self.model_type == "distilbert":
            model_path = "./models/best_distilbert_formality"
            from transformers import AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # Dùng pipeline custom để tự động strip token_type_ids
            self.classifier = _DistilBertPipeline(
                model=model,
                tokenizer=tokenizer,
            )

        elif self.model_type == "fasttext":
            if fasttext is None:
                raise ImportError(
                    "FastText chua duoc cai trong moi truong nay. "
                    "Hay dung RoBERTa/DistilBERT hoac cai fasttext (Python <= 3.12)."
                )
            model_path = "./models/best_fasttext_formality.bin"
            self.classifier = fasttext.load_model(model_path)

    def predict(self, text):
        # ── Transformer (RoBERTa / DistilBERT) ────────────────────────────────
        if self.model_type in ["roberta", "distilbert"]:
            raw_results = self.classifier(text, top_k=None)
            # Chuẩn hóa về list[dict] dù là 1 câu hay batch
            if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
                all_results = raw_results[0]
            else:
                all_results = raw_results

            top2_scores = []
            for result in all_results:
                mapped_label = "Formal" if result["label"] == "LABEL_1" else "Informal"
                top2_scores.append((mapped_label, result["score"] * 100))
            top2_scores.sort(key=lambda item: item[1], reverse=True)

            label = top2_scores[0][0]
            score = top2_scores[0][1]

        # ── FastText ──────────────────────────────────────────────────────────
        elif self.model_type == "fasttext":
            clean_text = text.replace("\n", " ")
            prediction = self.classifier.predict(clean_text)

            raw_label = prediction[0][0]
            label = "Formal" if raw_label == "__label__1" else "Informal"
            prob = float(prediction[1][0])
            prob = max(0.0, min(1.0, prob))
            score = prob * 100
            other_label = "Informal" if label == "Formal" else "Formal"
            top2_scores = [(label, score), (other_label, 100 - score)]

        return label, score, top2_scores