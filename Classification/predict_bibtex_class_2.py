import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
import bibtexparser
import os
import argparse


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AlbertClassifier(torch.nn.Module):
    def __init__(self, bert_model):
        super(AlbertClassifier, self).__init__()
        self.bert_model = bert_model

    def forward(self, token_ids, attention_mask=None):
        outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_mask)
        return outputs.logits


def load_test_bibtex(bibtex_path, tokenizer, max_length=512):
    """Load the test BibTeX file, extract entries and preprocessed features."""
    all_entries = []
    texts = []
    masks = []

    if os.path.exists(bibtex_path):
        with open(bibtex_path, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        for entry in bib_database.entries:
            # Save original entry
            all_entries.append(entry)

            # Get title and abstract (if they exist)
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')

            # Combine title and abstract as input text
            content = f"{title}\n{abstract}".strip()

            if content:
                # Encode content with tokenizer
                encoded = tokenizer(
                    content,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
                texts.append(encoded['input_ids'])
                masks.append(encoded['attention_mask'])
            else:
                # If no content, add an empty encoding
                # This case should be rare, but added for robustness
                empty_encoded = tokenizer(
                    " ",
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
                texts.append(empty_encoded['input_ids'])
                masks.append(empty_encoded['attention_mask'])

    return all_entries, texts, masks


def predict_and_save(model, tokenizer, test_bibtex_path, output_paths, device, batch_size=4, max_length=256):
    """Predict test data categories and save to two different BibTeX files."""
    model.eval()

    # Load test data
    print(f"Loading test data: {test_bibtex_path}")
    entries, texts, masks = load_test_bibtex(test_bibtex_path, tokenizer, max_length)
    print(f"Loaded {len(entries)} BibTeX records")

    # Prediction
    predictions = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_masks = masks[i:i + batch_size]

            # Convert to tensors and move to device
            batch_texts_tensor = torch.tensor(batch_texts).to(device)
            batch_masks_tensor = torch.tensor(batch_masks).to(device)

            # Predict
            outputs = model(batch_texts_tensor, attention_mask=batch_masks_tensor)
            _, predicted = torch.max(outputs, 1)

            # Collect predicted labels
            predictions.extend(predicted.cpu().numpy())

    # Assign entries to different categories based on predictions
    yes_entries = []
    no_entries = []

    for entry, pred in zip(entries, predictions):
        if pred == 1:  # Predicted as "YES" category
            yes_entries.append(entry)
        else:  # pred == 0, predicted as "NO" category
            no_entries.append(entry)

    # Create two BibTeX databases
    yes_db = bibtexparser.bibdatabase.BibDatabase()
    yes_db.entries = yes_entries

    no_db = bibtexparser.bibdatabase.BibDatabase()
    no_db.entries = no_entries

    # Save to files
    writer = bibtexparser.bwriter.BibTexWriter()

    with open(output_paths[0], 'w', encoding='utf-8') as f:
        f.write(writer.write(yes_db))

    with open(output_paths[1], 'w', encoding='utf-8') as f:
        f.write(writer.write(no_db))

    print("Classification results saved:")
    print(f"- YES category: {len(yes_entries)} entries -> {output_paths[0]}")
    print(f"- NO category: {len(no_entries)} entries -> {output_paths[1]}")


def main():
    parser = argparse.ArgumentParser(description='Predict BibTeX classification with a trained model')
    parser.add_argument('--model_path', type=str, default="path to best_model_Beye_class_2_scibert.pt",
                        help='Path to the trained model weights')
    parser.add_argument('--pretrained_model', type=str, default="path to scibert",
                        help='Path or name of the pretrained model')
    parser.add_argument('--test_data', type=str, default='path to evaluated test_data.bib',
                        help='Path to the BibTeX file containing test data')
    parser.add_argument('--output_dir', type=str, default='path to output',
                        help='Directory to save the output files')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    args = parser.parse_args()

    # Set output file paths - two files
    output_paths = [
        os.path.join(args.output_dir, "totel_yes.bib"),
        os.path.join(args.output_dir, "totel_no.bib")
    ]

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    set_seed()

    # Load tokenizer and model configuration
    print(f"Loading pretrained model: {args.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    config = AutoConfig.from_pretrained(args.pretrained_model)
    config.num_labels = 2  # Two-class classification

    # Load model architecture
    bert_model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
    classifier = AlbertClassifier(bert_model)

    # Load trained model weights
    print(f"Loading model weights: {args.model_path}")
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)

    # Run prediction and save results
    predict_and_save(
        model=classifier,
        tokenizer=tokenizer,
        test_bibtex_path=args.test_data,
        output_paths=output_paths,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()