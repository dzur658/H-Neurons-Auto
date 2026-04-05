from transformers.models.modernbert.modular_modernbert import ModernBertForQuestionAnswering
from transformers import AutoTokenizer, pipeline

class QnAModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertForQuestionAnswering.from_pretrained(model_name).cuda()
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

    def predict(self, question: str, context: str) -> dict:
        result = self.qa_pipeline(question=question, context=context)

        # return full result
        return result