import transformers
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from setup import get_model_params

""" testing with BERT """

if __name__ == "__main__":
    # setup
    model_name = "vinai/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(f"Model params: {get_model_params(model)}")  #