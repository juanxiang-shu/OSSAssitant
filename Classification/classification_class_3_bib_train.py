import optuna
from optuna.samplers import TPESampler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
import bibtexparser
from sklearn.metrics import classification_report, confusion_matrix
import os
import torch.nn.functional as F
import logging
import pandas as pd
from datetime import datetime
import psutil
import traceback
import sys
import gc
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import argparse
import json


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Configure logging
def setup_logging():
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"hpo_log_class_3_scibert_{current_time}.txt")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    return log_dir, current_time


# Load data from BibTeX files - 3-class classification
def load_bibtex_data(ppt_path, ft_path, review_path, tokenizer, max_length=512):
    """Load data from three BibTeX files, corresponding to three classes."""
    texts = []
    masks = []
    labels = []

    # Load PPT data (label = 0)
    if os.path.exists(ppt_path):
        with open(ppt_path, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        for entry in bib_database.entries:
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')

            content = f"{title}\n{abstract}".strip()

            if content:
                encoded = tokenizer(
                    content,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
                texts.append(encoded['input_ids'])
                masks.append(encoded['attention_mask'])
                labels.append(0)

    # Load FT data (label = 1)
    if os.path.exists(ft_path):
        with open(ft_path, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        for entry in bib_database.entries:
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')

            content = f"{title}\n{abstract}".strip()

            if content:
                encoded = tokenizer(
                    content,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
                texts.append(encoded['input_ids'])
                masks.append(encoded['attention_mask'])
                labels.append(1)

    # Load Review data (label = 2)
    if os.path.exists(review_path):
        with open(review_path, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        for entry in bib_database.entries:
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')

            content = f"{title}\n{abstract}".strip()

            if content:
                encoded = tokenizer(
                    content,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
                texts.append(encoded['input_ids'])
                masks.append(encoded['attention_mask'])
                labels.append(2)

    logging.info(f"Loading data statistics: PPT={labels.count(0)}, FT={labels.count(1)}, Review={labels.count(2)}")
    print(f"Loading data statistics: PPT={labels.count(0)}, FT={labels.count(1)}, Review={labels.count(2)}")
    return texts, masks, labels


class AlbertClassifier(torch.nn.Module):
    def __init__(self, bert_model):
        super(AlbertClassifier, self).__init__()
        self.bert_model = bert_model

    def forward(self, token_ids, attention_mask=None):
        outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_mask)
        return outputs.logits


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DataGen(Dataset):
    def __init__(self, data, masks, label):
        self.data = data
        self.masks = masks
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            np.array(self.data[index]),
            np.array(self.masks[index]),
            np.array(self.label[index], dtype=np.int64)
        )


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for token_ids, masks, labels in dataloader:
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(token_ids, attention_mask=masks)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate_with_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for token_ids, masks, labels in dataloader:
            token_ids = token_ids.to(device)
            masks = masks.to(device)

            outputs = model(token_ids, attention_mask=masks)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = ['PPT', 'FT', 'Review']  # Three-class category names
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)

    return conf_matrix, report


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10,
                early_stop_patience=5):
    best_val_acc = 0.0
    no_improve_epochs = 0

    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for step, (token_ids, masks, labels) in enumerate(train_dataloader):
                try:
                    token_ids = token_ids.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)

                    outputs = model(token_ids, attention_mask=masks)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    if (step + 1) % 10 == 0:
                        logging.info(
                            f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                        )

                        # Periodically check memory usage
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                            logging.info(
                                f"GPU Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB"
                            )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logging.error(f"Out of memory in training batch. Error: {str(e)}")
                        print(f"Out of memory in training batch. Error: {str(e)}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Skip this batch and continue training
                        continue
                    else:
                        raise e

            train_accuracy = train_correct / train_total if train_total > 0 else 0
            avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0

            # Validation phase; catch potential errors
            try:
                val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)

                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= early_stop_patience:
                        logging.info(f"Early Stop: No improvement after {early_stop_patience} rounds")
                        print(f"Early Stop: No improvement after {early_stop_patience} rounds")
                        break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.error(f"Out of memory during validation. Error: {str(e)}")
                    print(f"Out of memory during validation. Error: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # We might need to reduce batch size or stop earlier
                    no_improve_epochs += 1
                else:
                    raise e

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error during training: {str(e)}")
        print(traceback.format_exc())

    return best_val_acc


def objective(trial, data_paths, model_path, n_folds=5):
    # Hyperparameter optimization objective
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    try:
        # Monitor memory usage
        process = psutil.Process()
        logging.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # Hyperparameter search space
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        focal_gamma = trial.suggest_int("focal_gamma", 1, 7)
        max_length = trial.suggest_categorical("max_length", [128, 256, 384, 512])
        early_stop_patience = trial.suggest_int("early_stop_patience", 3, 10)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load BibTeX data
        ppt_path, ft_path, review_path = data_paths
        texts, masks, labels = load_bibtex_data(ppt_path, ft_path, review_path, tokenizer, max_length=max_length)

        # K-fold object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Store metrics for each fold
        fold_metrics = []

        # Index array for splitting
        indices = np.arange(len(texts))

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
            logging.info(f"\n{'=' * 50}")
            logging.info(f"Starting Fold {fold + 1}/{n_folds} for trial {trial.number}")
            print(f"\n{'=' * 50}")
            print(f"Starting Fold {fold + 1}/{n_folds} for trial {trial.number}")

            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logging.info(f"Memory usage before fold {fold + 1}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            try:
                # Split train/val within this fold
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=0.15,
                    random_state=42,
                    stratify=[labels[i] for i in train_val_idx]
                )

                # Build datasets
                X_train = [texts[i] for i in train_idx]
                X_train_masks = [masks[i] for i in train_idx]
                y_train = [labels[i] for i in train_idx]

                X_val = [texts[i] for i in val_idx]
                X_val_masks = [masks[i] for i in val_idx]
                y_val = [labels[i] for i in val_idx]

                X_test = [texts[i] for i in test_idx]
                X_test_masks = [masks[i] for i in test_idx]
                y_test = [labels[i] for i in test_idx]

                train_dataset = DataGen(X_train, X_train_masks, y_train)
                val_dataset = DataGen(X_val, X_val_masks, y_val)
                test_dataset = DataGen(X_test, X_test_masks, y_test)

                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

                # Initialize a new model for 3-class classification
                model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
                classifier = AlbertClassifier(model).to(device)

                # Compute class weights (3-class)
                class_counts = [y_train.count(0), y_train.count(1), y_train.count(2)]

                if sum(class_counts) > 0:
                    class_weights = [
                        len(y_train) / (3 * count) if count > 0 else 1.0
                        for count in class_counts
                    ]
                    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                else:
                    class_weights = None

                # Loss and optimizer
                criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
                optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

                # Train model
                train_model(
                    classifier,
                    train_dataloader,
                    val_dataloader,
                    criterion,
                    optimizer,
                    device,
                    num_epochs=10,
                    early_stop_patience=early_stop_patience
                )

                # Evaluate on test set
                test_loss, test_accuracy = evaluate(classifier, test_dataloader, criterion, device)

                # Detailed evaluation metrics
                _, report_dict = evaluate_with_metrics(classifier, test_dataloader, device)

                # Store metrics
                fold_metrics.append({
                    'test_accuracy': test_accuracy,
                    'f1_macro': report_dict['macro avg']['f1-score'],
                    'precision_macro': report_dict['macro avg']['precision'],
                    'recall_macro': report_dict['macro avg']['recall']
                })

                # Free memory
                del model, classifier, criterion, optimizer
                del train_dataset, val_dataset, test_dataset
                del train_dataloader, val_dataloader, test_dataloader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logging.error(f"Error in fold {fold + 1}: {str(e)}")
                logging.error(traceback.format_exc())
                print(f"Error in fold {fold + 1}: {str(e)}")
                print(traceback.format_exc())
                continue

        # If all folds failed, return a penalty
        if not fold_metrics:
            logging.error("All folds failed, returning penalty value")
            print("All folds failed, returning penalty value")
            return -1.0

        # Average metrics across folds
        avg_metrics = {k: np.mean([fold[k] for fold in fold_metrics]) for k in fold_metrics[0].keys()}

        # Store metrics to trial user attributes
        for key, value in avg_metrics.items():
            trial.set_user_attr(key, value)

        # Main optimization metric
        return avg_metrics['f1_macro']

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Trial {trial.number} failed with error: {str(e)}")
        print(traceback.format_exc())
        return -1.0


def run_bayesian_optimization(data_paths, model_path, n_trials=20, n_folds=5):
    log_dir, current_time = setup_logging()

    # Create a study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"bibtex_classification_hpo_{current_time}"
    )

    # Run optimization
    study.optimize(lambda trial: objective(trial, data_paths, model_path, n_folds), n_trials=n_trials)

    # Best results
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    logging.info("\n" + "=" * 50)
    logging.info("Hyperparameter Optimization Results:")
    logging.info(f"Best F1-Macro: {best_value:.4f}")
    logging.info("Best Parameters:")
    for param, value in best_params.items():
        logging.info(f"  - {param}: {value}")

    logging.info("Best Trial Metrics:")
    for key in ['test_accuracy', 'precision_macro', 'recall_macro']:
        logging.info(f"  - {key}: {best_trial.user_attrs[key]:.4f}")

    print("\n" + "=" * 50)
    print("Hyperparameter Optimization Results:")
    print(f"Best F1-Macro: {best_value:.4f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")

    # Save trials table
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(log_dir, f"hpo_trials_class_3_scibert_{current_time}.csv"), index=False)

    # Save params + user_attrs + value for each trial
    user_attrs_df = pd.DataFrame([
        {
            **t.params,
            **{f"user_attr_{k}": v for k, v in t.user_attrs.items()},
            "value": t.value
        }
        for t in study.trials
    ])
    user_attrs_df.to_csv(os.path.join(log_dir, f"hpo_trial_metrics_class_3_scibert_{current_time}.csv"), index=False)

    return best_params


# Train a final model with the best hyperparameters and evaluate
def train_final_model(best_params, data_paths, model_path, log_dir, current_time):
    """
    Train the full model with best hyperparameters and evaluate.
    """
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Unpack best hyperparameters
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    focal_gamma = best_params["focal_gamma"]
    max_length = best_params["max_length"]
    early_stop_patience = best_params["early_stop_patience"]

    logging.info("\n" + "=" * 50)
    logging.info("Starting Final Training with Best Parameters:")
    for param, value in best_params.items():
        logging.info(f"  - {param}: {value}")
    print("\n" + "=" * 50)
    print("Starting Final Training with Best Parameters:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load BibTeX data
    ppt_path, ft_path, review_path = data_paths
    texts, masks, labels = load_bibtex_data(ppt_path, ft_path, review_path, tokenizer, max_length=max_length)

    # Split into train and test
    train_idx, test_idx = train_test_split(
        np.arange(len(texts)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Further split train into train and validation
    train_idx_subset, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=[labels[i] for i in train_idx]
    )

    # Build datasets
    X_train = [texts[i] for i in train_idx_subset]
    X_train_masks = [masks[i] for i in train_idx_subset]
    y_train = [labels[i] for i in train_idx_subset]

    X_val = [texts[i] for i in val_idx]
    X_val_masks = [masks[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    X_test = [texts[i] for i in test_idx]
    X_test_masks = [masks[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")
    logging.info(f"Test set size: {len(X_test)}")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Create dataloaders
    train_dataset = DataGen(X_train, X_train_masks, y_train)
    val_dataset = DataGen(X_val, X_val_masks, y_val)
    test_dataset = DataGen(X_test, X_test_masks, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize a new 3-class model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    classifier = AlbertClassifier(model).to(device)

    # Compute class weights
    class_counts = [y_train.count(0), y_train.count(1), y_train.count(2)]
    logging.info(
        f"Number of samples per category: PPT={class_counts[0]}, FT={class_counts[1]}, Review={class_counts[2]}"
    )
    print(
        f"Number of samples per category: PPT={class_counts[0]}, FT={class_counts[1]}, Review={class_counts[2]}"
    )

    if sum(class_counts) > 0:
        class_weights = [
            len(y_train) / (3 * count) if count > 0 else 1.0
            for count in class_counts
        ]
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"Category weight: {class_weights}")
        print(f"Category weight: {class_weights}")
    else:
        class_weights = None

    # Define loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler (optional improvement)
    total_steps = len(train_dataloader) * 10  # up to 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    # Training loop
    final_model_path = os.path.join(log_dir, f"best_model_Beye_class_3_scibert_{current_time}.pt")

    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(10):  # up to 10 epochs
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (token_ids, masks, labels_batch) in enumerate(train_dataloader):
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels_batch = labels_batch.to(device)

            outputs = classifier(token_ids, attention_mask=masks)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()

            if (step + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch + 1}/10], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )
                print(
                    f"Epoch [{epoch + 1}/10], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

        train_accuracy = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(classifier, val_dataloader, criterion, device)

        logging.info(
            f"Epoch {epoch + 1}/10, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch + 1}/10, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save best model checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improve_epochs = 0

            torch.save(classifier.state_dict(), final_model_path)
            logging.info(f"Model saved successfully! Validation accuracy: {best_val_acc:.4f}")
            print(f"Model saved successfully! Validation accuracy: {best_val_acc:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logging.info(f"Early Stop: No improvement after {early_stop_patience} rounds")
                print(f"Early Stop: No improvement after {early_stop_patience} rounds")
                break

    # Load best model and evaluate on test set
    classifier.load_state_dict(torch.load(final_model_path))

    test_loss, test_accuracy = evaluate(classifier, test_dataloader, criterion, device)
    logging.info(f"Final Test results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    print(f"Final Test results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Detailed evaluation
    conf_matrix, report_dict = evaluate_with_metrics(classifier, test_dataloader, device)

    logging.info("Final Confusion Matrix:")
    logging.info(f"\n{conf_matrix}")
    print("Final Confusion Matrix:")
    print(f"\n{conf_matrix}")

    # Classification report and CSV output
    try:
        # Predict all test data
        predictions = []
        classifier.eval()
        with torch.no_grad():
            for i in range(len(X_test)):
                encoded_tensor = torch.tensor([X_test[i]]).to(device)
                mask_tensor = torch.tensor([X_test_masks[i]]).to(device)
                output = classifier(encoded_tensor, attention_mask=mask_tensor)
                prediction = torch.argmax(output, dim=1).item()
                predictions.append(prediction)

        # Generate classification report
        report_str = classification_report(y_test, predictions, target_names=['PPT', 'FT', 'Review'])
        logging.info("Final Classification report:")
        logging.info(f"\n{report_str}")
        print("Final Classification report:")
        print(f"\n{report_str}")

        # Ensure category names exist in report_dict
        report_dict['PPT'] = report_dict.get('PPT', report_dict.get('0', {'f1-score': 0.0}))
        report_dict['FT'] = report_dict.get('FT', report_dict.get('1', {'f1-score': 0.0}))
        report_dict['Review'] = report_dict.get('Review', report_dict.get('2', {'f1-score': 0.0}))

        # Save final results to CSV
        results_df = pd.DataFrame({
            'metric': [
                'test_accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                'f1_PPT', 'f1_FT', 'f1_Review'
            ],
            'value': [
                test_accuracy,
                report_dict['macro avg']['f1-score'],
                report_dict['macro avg']['precision'],
                report_dict['macro avg']['recall'],
                report_dict['PPT']['f1-score'],
                report_dict['FT']['f1-score'],
                report_dict['Review']['f1-score']
            ]
        })
        results_df.to_csv(
            os.path.join(log_dir, f"final_results_class_3_scibert_{current_time}.csv"),
            index=False
        )

    except Exception as e:
        logging.error(f"Error in generating classification report: {str(e)}")
        print(f"Error in generating classification report: {str(e)}")

    return classifier, test_accuracy, report_dict


def run_pretrained_baseline_multiclass(
    data_paths,
    model_path,
    log_dir,
    current_time,
    batch_size=8,
    max_length=512,
    seed=42
):
    """
    3-class baseline: load a pretrained model (without fine-tuning) and evaluate on a fixed split.
    Outputs: baseline metrics including macro F1, per-class F1, balanced accuracy,
    ROC-AUC(macro), PR-AUC(macro), MCC, test_loss, and n_test.
    """
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ppt_path, ft_path, review_path = data_paths
    texts, masks, labels = load_bibtex_data(ppt_path, ft_path, review_path, tokenizer, max_length=max_length)

    # Same split strategy as main training
    all_idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    train_idx_subset, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=[labels[i] for i in train_idx]
    )

    def slice_set(idxs):
        return [texts[i] for i in idxs], [masks[i] for i in idxs], [labels[i] for i in idxs]

    X_train, X_train_masks, y_train = slice_set(train_idx_subset)
    X_val, X_val_masks, y_val = slice_set(val_idx)
    X_test, X_test_masks, y_test = slice_set(test_idx)

    train_loader = DataLoader(DataGen(X_train, X_train_masks, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(DataGen(X_val, X_val_masks, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DataGen(X_test, X_test_masks, y_test), batch_size=batch_size, shuffle=False)

    # Pretrained, unfine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    classifier = AlbertClassifier(model).to(device)
    classifier.eval()

    # Loss for reporting only (no training)
    class_counts = [y_train.count(0), y_train.count(1), y_train.count(2)]
    class_weights = None
    if sum(class_counts) > 0:
        cw = [len(y_train) / (3 * c) if c > 0 else 1.0 for c in class_counts]
        class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Evaluate on train/val (for consistency, but not used)
    _ = evaluate(classifier, train_loader, criterion, device)
    _ = evaluate(classifier, val_loader, criterion, device)

    # Test evaluation
    all_labels, all_preds, all_probs = [], [], []
    test_loss, total, correct = 0.0, 0, 0

    with torch.no_grad():
        for token_ids, masks, labels_batch in test_loader:
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            labels_batch = labels_batch.to(device)

            logits = classifier(token_ids, attention_mask=masks)
            loss = criterion(logits, labels_batch)
            test_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

            all_labels.extend(labels_batch.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    test_loss = test_loss / max(1, len(test_loader))
    test_acc = correct / max(1, total)
    all_probs = np.vstack(all_probs)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Standard classification report
    class_names = ['PPT', 'FT', 'Review']
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    prec_macro = report['macro avg']['precision']
    rec_macro = report['macro avg']['recall']
    f1_PPT = report['PPT']['f1-score']
    f1_FT = report['FT']['f1-score']
    f1_Review = report['Review']['f1-score']

    # Extra metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc_macro = roc_auc_score(y_true, all_probs, multi_class='ovr', average='macro')
    Y_bin = label_binarize(y_true, classes=[0, 1, 2])
    pr_auc_macro = average_precision_score(Y_bin, all_probs, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)

    logging.info(
        "[Baseline-3C] Test Loss: %.4f | Acc: %.4f | F1_macro: %.4f | P_macro: %.4f | R_macro: %.4f | "
        "F1_PPT: %.4f | F1_FT: %.4f | F1_Review: %.4f | BalancedAcc: %.4f | ROC-AUC(macro): %.4f | "
        "PR-AUC(macro): %.4f | MCC: %.4f",
        test_loss, test_acc, f1_macro, prec_macro, rec_macro,
        f1_PPT, f1_FT, f1_Review, bal_acc, roc_auc_macro, pr_auc_macro, mcc
    )

    print(
        "[Baseline-3C] Test Loss: %.4f | Acc: %.4f | F1_macro: %.4f | P_macro: %.4f | R_macro: %.4f | "
        "F1_PPT: %.4f | F1_FT: %.4f | F1_Review: %.4f | BalancedAcc: %.4f | ROC-AUC(macro): %.4f | "
        "PR-AUC(macro): %.4f | MCC: %.4f"
        % (
            test_loss, test_acc, f1_macro, prec_macro, rec_macro,
            f1_PPT, f1_FT, f1_Review, bal_acc, roc_auc_macro, pr_auc_macro, mcc
        )
    )

    # Export CSV with baseline metrics
    baseline_row = {
        "user_attrs_f1_macro": f1_macro,
        "user_attrs_precision_macro": prec_macro,
        "user_attrs_recall_macro": rec_macro,
        "user_attrs_test_accuracy": test_acc,
        "f1_PPT": f1_PPT,
        "f1_FT": f1_FT,
        "f1_Review": f1_Review,
        "balanced_accuracy": bal_acc,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "mcc": mcc,
        "test_loss": test_loss,
        "n_test": int(len(y_true))
    }
    df = pd.DataFrame([baseline_row])
    out_csv = os.path.join(log_dir, f"baseline_pretrained_scibert_3class_metrics_{current_time}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Baseline-3C] saved to: {out_csv}")

    return baseline_row, out_csv


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--skip_hpo",
            action="store_true",
            help="Skip Bayesian optimization, directly load best_params.json"
        )
        parser.add_argument(
            "--baseline_only",
            action="store_true",
            help="Only evaluate pretrained (untrained) SciBERT baseline on fixed split"
        )
        args = parser.parse_args()

        # Set BibTeX file paths
        ppt_path = "path your data/ppt.bib"
        ft_path = "path your data/ft.bib"
        review_path = "path your data/review.bib"

        # Set model path
        model_path = "path your models/scibert"

        # Setup logging
        log_dir, current_time = setup_logging()

        # Override log filename (optional)
        log_filename = os.path.join(log_dir, f"hpo_log_class_3_scibert_{current_time}.txt")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s"
        )

        set_seed(42)

        # Add system information logs
        import platform
        total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)

        logging.info(f"System: {platform.system()} {platform.version()}")
        logging.info(f"Python version: {platform.python_version()}")
        logging.info(f"Total memory: {total_memory:.2f} GB")
        logging.info(f"Available memory: {available_memory:.2f} GB")

        print(f"System: {platform.system()} {platform.version()}")
        print(f"Python version: {platform.python_version()}")
        print(f"Total memory: {total_memory:.2f} GB")
        print(f"Available memory: {available_memory:.2f} GB")

        if torch.cuda.is_available():
            logging.info("CUDA available: True")
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB"
            )

            print("CUDA available: True")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB"
            )
        else:
            logging.info("CUDA not available")
            print("CUDA not available")

        # Baseline-only mode (no HPO, no training)
        if args.baseline_only:
            print("Only evaluating pretrained (untrained) SciBERT baseline for 3-class…")
            logging.info("Run pretrained baseline only (no HPO, no training) for 3-class.")
            baseline_metrics, out_csv = run_pretrained_baseline_multiclass(
                data_paths=(ppt_path, ft_path, review_path),
                model_path=model_path,
                log_dir=log_dir,
                current_time=current_time,
                batch_size=8,
                max_length=512
            )
            print("Baseline metrics:", baseline_metrics)
            print("Done.")
            sys.exit(0)

        # Decide whether to run Bayesian optimization
        if not args.skip_hpo:
            logging.info("Starting Bayesian optimization...")
            print("Starting Bayesian optimization...")

            best_params = run_bayesian_optimization(
                data_paths=(ppt_path, ft_path, review_path),
                model_path=model_path,
                n_trials=20,
                n_folds=5
            )

            best_params_path = os.path.join(log_dir, "best_params_class_3_scibert_.json")
            with open(best_params_path, "w") as fout:
                json.dump(best_params, fout, indent=2)

            logging.info(f"Best parameters saved to: {best_params_path}")
            print(f"Best parameters saved to: {best_params_path}")
        else:
            logging.info("Skipping Bayesian optimization, loading best parameters directly...")
            print("Skipping Bayesian optimization, loading best parameters directly...")

            best_params_paths = [
                "best_params.json",
                os.path.join(log_dir, "best_params.json"),
                "./log/best_params.json"
            ]

            best_params = None
            for path in best_params_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as fin:
                            best_params = json.load(fin)
                        logging.info(f"Successfully loaded best parameters from {path}")
                        print(f"Successfully loaded best parameters from {path}")
                        break
                    except Exception as e:
                        logging.warning(f"Failed to load parameters from {path}: {str(e)}")
                        print(f"Failed to load parameters from {path}: {str(e)}")

            if best_params is None:
                logging.error("Cannot find best_params.json file!")
                print("Error: Cannot find best_params.json file!")
                print("Please ensure the file exists in one of the following locations:")
                for path in best_params_paths:
                    print(f"  - {path}")
                sys.exit(1)

            logging.info("Loaded best parameters:")
            print("Loaded best parameters:")
            for param, value in best_params.items():
                logging.info(f"  - {param}: {value}")
                print(f"  - {param}: {value}")

        # Train and evaluate final model with best hyperparameters
        logging.info("Starting to train final model with best parameters...")
        print("Starting to train final model with best parameters...")

        final_model, final_accuracy, final_report = train_final_model(
            best_params,
            data_paths=(ppt_path, ft_path, review_path),
            model_path=model_path,
            log_dir=log_dir,
            current_time=current_time
        )

        logging.info("\n" + "=" * 50)
        logging.info("Complete Workflow Summary:")
        logging.info("Best Parameters:")
        for param, value in best_params.items():
            logging.info(f"  - {param}: {value}")
        logging.info(f"Final Model Test Accuracy: {final_accuracy:.4f}")
        logging.info(f"Final Model Macro F1: {final_report['macro avg']['f1-score']:.4f}")

        print("\n" + "=" * 50)
        print("Complete Workflow Summary:")
        print("Best Parameters:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        print(f"Final Model Test Accuracy: {final_accuracy:.4f}")
        print(f"Final Model Macro F1: {final_report['macro avg']['f1-score']:.4f}")

        # Save summary
        summary_path = os.path.join(log_dir, f"workflow_summary_class_3_scibert_{current_time}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Complete Workflow Summary:\n")
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"  - {param}: {value}\n")
            f.write(f"Final Model Test Accuracy: {final_accuracy:.4f}\n")
            f.write(f"Final Model Macro F1: {final_report['macro avg']['f1-score']:.4f}\n")

        logging.info(f"Workflow summary saved to: {summary_path}")
        print(f"Workflow summary saved to: {summary_path}")

    except Exception as e:
        logging.error(f"Critical error in main function: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Critical error in main function: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)