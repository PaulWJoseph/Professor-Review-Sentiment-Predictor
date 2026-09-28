# Code changed to Python. Originally from colab.research.google.com in Jupyter Notebook.
# Jupyter Notebook: https://colab.research.google.com/drive/1h8vS4ESBEJQEQFL3kjXyUuVSPS1Gwviw?usp=sharing

import random
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import planetterp
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def fetch_professor_reviews(professors_list):
    """Fetch reviews from PlanetTerp API for specified professors."""
    reviews = []
    for professor in professors_list:
        prof = planetterp.professor(name=professor, reviews=True)
        for review in prof.get("reviews", []):
            if review.get("review") and review.get("rating"):
                reviews.append(
                    {
                        "text": review["review"],
                        "label": int(review["rating"]) - 1,  # Convert 1-5 to 0-4
                        "professor": professor,
                    }
                )
    return reviews


def prepare_dataloaders(reviews, checkpoint="bert-base-uncased", batch_size=16):
    """Tokenize dataset and prepare PyTorch DataLoaders."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    dataset = Dataset.from_list(reviews)
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

    keep_test_dataset = dataset["test"]

    tokenized_data = dataset.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns(["text", "professor"])
    tokenized_data = tokenized_data.rename_column("label", "labels")
    tokenized_data.set_format("pt")

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_data["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = torch.utils.data.DataLoader(
        tokenized_data["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader, test_dataloader, keep_test_dataset, tokenizer


def train_model(model, train_dataloader, epochs=7, lr=3e-5, device="cpu"):
    """Fine-tune BERT model."""
    num_training_steps = epochs * len(train_dataloader)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.to(device)
    progress_bar = tqdm(range(num_training_steps), desc="Training")

    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return model


def evaluate_model(model, test_dataloader, keep_test_dataset, device="cpu"):
    """Evaluate trained model and return predictions and accuracy metrics."""
    preds = []
    model.eval()

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        yhat = torch.argmax(logits, dim=-1)
        preds.append(yhat.cpu())

    prediction_list = torch.cat(preds).tolist()
    stars_predicted = [star + 1 for star in prediction_list]

    final_results = []
    for index, star in enumerate(stars_predicted):
        curr_review = keep_test_dataset[index]
        final_results.append(
            {
                "professor": curr_review["professor"],
                "review": curr_review["text"],
                "rating": curr_review["label"] + 1,
                "predicted_rating": star,
            }
        )

    exact_matches = sum(
        1 for r in final_results if r["rating"] == r["predicted_rating"]
    )
    within_one = sum(
        1 for r in final_results if abs(r["rating"] - r["predicted_rating"]) <= 1
    )
    total = len(final_results)

    exact_acc = exact_matches / total if total > 0 else 0
    within_one_acc = within_one / total if total > 0 else 0

    return final_results, exact_acc, within_one_acc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    professors = [
        "Cliff Bakalian",
        "Nelson Padua-Perez",
        "Fawzi Emad",
        "Jieun Yeon",
        "Samantha Kemper",
    ]

    print("Fetching reviews...")
    reviews = fetch_professor_reviews(professors)
    print(f"Total reviews retrieved: {len(reviews)}")

    checkpoint = "bert-base-uncased"
    train_dataloader, test_dataloader, keep_test_dataset, _ = prepare_dataloaders(
        reviews, checkpoint=checkpoint, batch_size=16
    )

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=5
    )

    print("Training model...")
    model = train_model(
        model, train_dataloader, epochs=7, lr=3e-5, device=device
    )

    print("Evaluating model...")
    results, exact_acc, within_one_acc = evaluate_model(
        model, test_dataloader, keep_test_dataset, device=device
    )

    print(f"\nExact Match Accuracy: {exact_acc:.4f}")
    print(f"Within +/- 1 Star Accuracy: {within_one_acc:.4f}\n")

    # Display misclassified examples (>1 star difference)
    print("--- Significant Misclassifications ---")
    for r in results:
        if abs(r["rating"] - r["predicted_rating"]) > 1:
            print(f"Professor: {r['professor']}")
            print(f"Review: {r['review']}")
            print(f"Actual: {r['rating']} | Predicted: {r['predicted_rating']}\n")


if __name__ == "__main__":
    main()