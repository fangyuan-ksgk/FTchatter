#################
##   __      _ ##
## o'')}____// ##
##  `_/      ) ##
##  (_(_/-(_/  ##
#################
##   OOC Data  ##
#################



# Prefix which Maria will use when it generates OOC response
# --- When we detect OOC behavior from Maria, we replace the prefix and let Maria continue from the following
ooc_patch = [
    "It almost feels like I am the sales here 🤷 Why should I consider FWD ",
    "Well, since you want to sell me insurance, can you ",
    "How come I feels liks a sales here 🤷 you are supposed to ",
    "I don't know whether insurance is right for me ",
    "Anyway Alex, why should I be interested in your product ",
    "It's funny that you are selling me insurance and not the other way around, "
]

##############################
## OOC Responses from Maria ##
##############################

ooc_responses = [
    "So, what do you think about our products? Have you heard of FWD before? 🙆",
    "So, what do you think about life insurance? Have you ever thought about it before?",
    "So, do you have any other questions about our policies? 🤔",
    "Well, let me tell you why it's great. We have flexible plans and affordable premiums. Plus, our customer service is top-notch. 😊",
    "Nice to meet you, I am Alex from FWD insurance",
    "Right, back to our insurance product, Maria.",
    "So I am here to introduce our insurance product to you",
    "How about you? Ever thought of getting one?",
    "Well, FWD is actually a life insurance company that offers flexible and comprehensive plans for Filipinos. We believe in making insurance accessible and easy to understand.",
]

########################################
## Response from Maria which is Okay  ##
########################################

ok_responses = [
    "Uh, so Sam told me that you are trying to get into investment",
]


##############################################
## Train A Embedder Model for OOC Detection ##
##############################################

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch 
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset 


class TextClassificationDataset_v2(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_em(train_dataset, test_dataset, num_epochs=2):
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Torch Dataset 
    train_dataset_torch = TextClassificationDataset_v2(train_dataset["text"], train_dataset["label"], tokenizer, max_length=128)
    test_dataset_torch = TextClassificationDataset_v2(test_dataset["text"], test_dataset["label"], tokenizer, max_length=128)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            texts = batch['text']
            labels = batch['label'].to(device)

            # Tokenize the texts
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                texts = batch['text']
                labels = batch['label'].to(device)

                # Tokenize the texts
                encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)

                outputs = model(input_ids)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Agent", "Customer"]))

    return tokenizer, model


def predict_em(sample_text, model, tokenizer, threshold_agent=0, threshold_customer=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tokenize the sample text
    encoded = tokenizer(sample_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()

    # Map the predicted class to the corresponding label
    label_map = {0: "Agent", 1: "Customer"}
    thres_map = {0: threshold_agent, 1: threshold_customer}
    if predicted_confidence < thres_map[predicted_class]:
        return "Not Sure", 0
    else:
        predicted_label = label_map[predicted_class]
        return predicted_label, predicted_confidence
