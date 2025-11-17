"""
Simplified Litigation Classifier with Multi-Model Comparison
Supports: FinBERT, BERT, ELECTRA
CSV Format: ,lt_no,yr,title,lt
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
import json
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class LitigationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LitigationClassifier:
    """Simplified multi-model classifier for insider trading detection"""
    
    MODELS = {
        'finbert': 'ProsusAI/finbert',
        'bert': 'bert-base-uncased',
        'electra': 'google/electra-base-discriminator'
    }
    
    INSIDER_KEYWORDS = [
        'insider trading', 'material nonpublic', 'tipping', 'tippee',
        'rule 10b-5', 'section 10(b)', 'misappropriation', 'securities fraud'
    ]
    
    def __init__(self, model_type='finbert', output_dir='output', max_length=512):
        """
        Args:
            model_type: 'finbert', 'bert', or 'electra'
            output_dir: Directory for saving results
            max_length: Max token length
        """
        self.model_type = model_type
        self.model_name = self.MODELS[model_type]
        self.output_dir = "output"
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized {model_type.upper()} classifier on {self.device}")
    
    def load_data(self, csv_path):
        """Load CSV with format: ,lt_no,yr,title,lt"""
        df = pd.read_csv(csv_path, index_col=0)
        
        # Combine title + text
        df['text'] = (df['title'].fillna('') + '. ' + df['lt'].fillna('')).str.strip()
        df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Auto-label based on keywords
        df['label'] = df['text'].apply(self._has_insider_keywords)
        
        # Extract company names
        df['company'] = df['text'].apply(self._extract_company)
        
        logger.info(f"Loaded {len(df)} cases | Insider: {df['label'].sum()} | Years: {df['yr'].min()}-{df['yr'].max()}")
        return df
    
    def _has_insider_keywords(self, text):
        """Check if text contains insider trading keywords"""
        text_lower = str(text).lower()
        return int(any(kw in text_lower for kw in self.INSIDER_KEYWORDS))
    
    def _extract_company(self, text):
        """Simple company name extraction"""
        match = re.search(r'(?:SEC charges|charges against)\s+([A-Z][A-Za-z\s&]+?)(?:\s+(?:and|with|for))', str(text))
        if match:
            return match.group(1).strip()
        match = re.search(r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})', str(text))
        return match.group(1).strip() if match else "Unknown"
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """Split data into train/val/test"""
        train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
        train, val = train_test_split(train_val, test_size=val_size, stratify=train_val['label'], random_state=42)
        
        train_dataset = LitigationDataset(train['text'].tolist(), train['label'].tolist(), self.tokenizer, self.max_length)
        val_dataset = LitigationDataset(val['text'].tolist(), val['label'].tolist(), self.tokenizer, self.max_length)
        test_dataset = LitigationDataset(test['text'].tolist(), test['label'].tolist(), self.tokenizer, self.max_length)
        
        logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train_dataset, val_dataset, test_dataset, test
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, lr=2e-5):
        """Train the model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, ignore_mismatched_sizes=True
        ).to(self.device)
    
        training_args = TrainingArguments(
            output_dir=self.output_dir + 'training',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            eval_strategy='epoch',
            save_strategy='epoch'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        trainer.train()
        self.model.save_pretrained(self.output_dir / 'model')
        self.tokenizer.save_pretrained(self.output_dir / 'model')
        logger.info(f"Model saved to {self.output_dir / 'model'}")
    
    def _compute_metrics(self, pred):
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        return {'accuracy': accuracy_score(labels, preds), 'precision': precision, 'recall': recall, 'f1': f1}
    
    def evaluate(self, test_dataset):
        """Evaluate model and return metrics"""
        if self.model is None:
            model_path = os.path.join(self.output_dir, 'model')
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                probs = torch.softmax(outputs.logits, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        acc = accuracy_score(all_labels, all_preds)
        
        metrics = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
        
        logger.info(f"\n{self.model_type.upper()} Results: Acc={acc:.3f} | P={precision:.3f} | R={recall:.3f} | F1={f1:.3f}")
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics, all_preds, all_probs
    
    def predict(self, df, batch_size=16):
        """Classify dataframe and add predictions"""
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir / 'model').to(self.device)
        
        self.model.eval()
        dataset = LitigationDataset(df['text'].tolist(), [0]*len(df), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds, all_probs = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                probs = torch.softmax(outputs.logits, dim=1)
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        df = df.copy()
        df[f'{self.model_type}_pred'] = all_preds
        df[f'{self.model_type}_prob'] = all_probs
        df[f'{self.model_type}_class'] = df[f'{self.model_type}_prob'].apply(
            lambda x: 'Definite Insider' if x >= 0.85 else ('Possible Insider' if x >= 0.50 else 'Not Insider')
        )
        
        return df
    
    def generate_report(self, df, output_file='report.xlsx'):
        """Generate classification report"""
        class_col = f'{self.model_type}_class'
        prob_col = f'{self.model_type}_prob'
        
        # Summary by classification
        summary = df[class_col].value_counts()
        logger.info(f"\n{self.model_type.upper()} Classification Summary:")
        for cat, count in summary.items():
            logger.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
        
        # Company aggregation
        company_summary = df.groupby('company').agg({
            class_col: lambda x: (x.isin(['Definite Insider', 'Possible Insider'])).sum(),
            prob_col: 'mean',
            'lt_no': 'count'
        }).reset_index()
        company_summary.columns = ['company', 'insider_cases', 'avg_prob', 'total_cases']
        company_summary = company_summary.sort_values('insider_cases', ascending=False)
        
        logger.info(f"\nTop 5 Companies (Insider Cases):")
        for _, row in company_summary.head(5).iterrows():
            logger.info(f"  {row['company']}: {row['insider_cases']} cases")
        
        # Save to Excel
        with pd.ExcelWriter(self.output_dir / output_file, engine='openpyxl') as writer:
            for category in ['Definite Insider', 'Possible Insider', 'Not Insider']:
                subset = df[df[class_col] == category][['lt_no', 'yr', 'company', 'title', prob_col]]
                subset.to_excel(writer, sheet_name=category[:31], index=False)
            company_summary.to_excel(writer, sheet_name='Company Summary', index=False)
        
        logger.info(f"Report saved: {self.output_dir / output_file}")
        return company_summary


class ModelComparison:
    """Compare multiple models on the same dataset"""
    
    def __init__(self, output_dir='comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def compare_models(self, csv_path, models=['finbert', 'bert', 'electra'], epochs=3):
        """Train and evaluate multiple models"""
        logger.info(f"\n{'='*80}\nCOMPARING MODELS: {', '.join([m.upper() for m in models])}\n{'='*80}")
        
        comparison_df = None
        
        for model_type in models:
            logger.info(f"\n--- Training {model_type.upper()} ---")
            
            classifier = LitigationClassifier(model_type, output_dir=str(self.output_dir))
            
            if comparison_df is None:
                df = classifier.load_data(csv_path)
                train_ds, val_ds, test_ds, test_df = classifier.prepare_data(df)
                comparison_df = test_df.copy()
            else:
                df = classifier.load_data(csv_path)
                train_ds, val_ds, test_ds, _ = classifier.prepare_data(df)
            
            # Train
            classifier.train(train_ds, val_ds, epochs=epochs)
            
            # Evaluate
            metrics, preds, probs = classifier.evaluate(test_ds)
            self.results[model_type] = metrics
            
            # Add predictions to comparison dataframe
            comparison_df[f'{model_type}_pred'] = preds
            comparison_df[f'{model_type}_prob'] = probs
            comparison_df[f'{model_type}_class'] = comparison_df[f'{model_type}_prob'].apply(
                lambda x: 'Definite' if x >= 0.85 else ('Possible' if x >= 0.50 else 'Not Insider')
            )
        
        # Generate comparison report
        self._print_comparison()
        self._save_comparison(comparison_df)
        
        return comparison_df
    
    def _print_comparison(self):
        """Print comparison table"""
        logger.info(f"\n{'='*80}\nMODEL COMPARISON RESULTS\n{'='*80}")
        logger.info(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        logger.info("-"*80)
        
        for model, metrics in self.results.items():
            logger.info(
                f"{model.upper():<12} "
                f"{metrics['accuracy']:<10.4f} "
                f"{metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} "
                f"{metrics['f1']:<10.4f}"
            )
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        logger.info(f"\nBest Model: {best_model[0].upper()} (F1={best_model[1]['f1']:.4f})")
    
    def _save_comparison(self, df):
        """Save comparison results"""
        # Save metrics
        with open(self.output_dir / 'comparison_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save predictions
        df.to_csv(self.output_dir / 'comparison_predictions.csv', index=False)
        
        # Excel with agreement analysis
        with pd.ExcelWriter(self.output_dir / 'model_comparison.xlsx', engine='openpyxl') as writer:
            metrics_df = pd.DataFrame(self.results).T
            metrics_df.to_excel(writer, sheet_name='Metrics')
            
            df.to_excel(writer, sheet_name='All Predictions', index=False)
            
            if len(self.results) > 1:
                model_cols = [f'{m}_class' for m in self.results.keys()]
                df['agreement'] = df[model_cols].apply(lambda row: len(set(row)), axis=1) == 1
                agreement_rate = df['agreement'].mean()
                
                agreement_df = pd.DataFrame({
                    'Metric': ['Agreement Rate', 'Total Cases', 'Agreed Cases'],
                    'Value': [f"{agreement_rate:.2%}", len(df), df['agreement'].sum()]
                })
                agreement_df.to_excel(writer, sheet_name='Agreement', index=False)
                
                logger.info(f"\nModel Agreement: {agreement_rate:.1%} of cases")
        
        logger.info(f"Comparison saved: {self.output_dir / 'model_comparison.xlsx'}")


def main():    
    classifier = LitigationClassifier(model_type='finbert')
    df = classifier.load_data('litigation_data\litigation.csv')
    train_ds, val_ds, test_ds, test_df = classifier.prepare_data(df)
    classifier.train(train_ds, val_ds, epochs=3)
    classifier.evaluate(test_ds)
    
    df_classified = classifier.predict(df)
    classifier.generate_report(df_classified, 'finbert_report.xlsx')
    
##    comparison = ModelComparison(output_dir='model_comparison')
 ##   comparison_df = comparison.compare_models(
 ##       csv_path='litigation_data.csv',
 ##       models=['finbert', 'bert', 'electra'],
  ##      epochs=3
  ##  )


if __name__ == "__main__":
    main()