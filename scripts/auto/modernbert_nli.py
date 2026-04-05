import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        # inference only, do not track gradients
        self.model.eval()

        # Normalize checkpoint-specific labels to stable judge keys.
        self.class_to_judge_key = {}
        self.unmapped_class_labels = {}
        id2label = self.model.config.id2label or {}
        for class_id, raw_label in id2label.items():
            class_id_int = int(class_id)
            judge_key = self._to_judge_key(str(raw_label))
            if judge_key is not None:
                self.class_to_judge_key[class_id_int] = judge_key
            else:
                self.unmapped_class_labels[class_id_int] = str(raw_label)

        label2id = self.model.config.label2id or {}
        for raw_label, class_id in label2id.items():
            class_id_int = int(class_id)
            if class_id_int in self.class_to_judge_key:
                continue

            judge_key = self._to_judge_key(str(raw_label))
            if judge_key is not None:
                self.class_to_judge_key[class_id_int] = judge_key
                self.unmapped_class_labels.pop(class_id_int, None)
            else:
                self.unmapped_class_labels[class_id_int] = str(raw_label)

    @staticmethod
    def _to_judge_key(raw_label: str):
        normalized = raw_label.strip().lower().replace("_", " ").replace("-", " ")

        if "entail" in normalized or "support" in normalized or normalized == "yes":
            return "entailment"
        if "contradiction" in normalized or "refute" in normalized or normalized == "no":
            return "contradiction"
        if "neutral" in normalized or "unknown" in normalized or "neither" in normalized:
            return "neutral"
        return None

    def predict(self, premise: str, hypothesis: str) -> dict:
        # package hypothesis
        pkg_hypothesis = f"The correct answer is: {hypothesis}"

        inputs = self.tokenizer(
            premise, 
            pkg_hypothesis, 
            return_tensors='pt',
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # prcess outputs
        logits = outputs.logits

        # convert logits to percents
        probs = F.softmax(logits, dim=-1)

        # grab highest percentage prediction
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item() * 100

        # map predicted class using the model's configured labels
        id2label = self.model.config.id2label or {}
        raw_label = id2label.get(predicted_class, str(predicted_class))
        judge_key = self.class_to_judge_key.get(predicted_class)
        if judge_key is None:
            raise ValueError(
                "Could not map predicted NLI class to a judge key. "
                f"model={self.model_name}, predicted_class={predicted_class}, raw_label={raw_label}, "
                f"id2label={id2label}, label2id={self.model.config.label2id}, "
                f"known_mapping={self.class_to_judge_key}, unmapped_labels={self.unmapped_class_labels}"
            )
        
        output_dict = {
            'predicted_class': predicted_class,
            'raw_label': raw_label,
            'judge_key': judge_key,
            'confidence': confidence,
        }
        return output_dict