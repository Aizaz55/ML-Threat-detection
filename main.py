import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

class CICIDSThreatDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def load_and_merge_csvs(self, folder_path, max_chunks=5, chunk_size=10000):
        """Load and merge multiple CSV files from CICIDS2017 dataset."""
        print("Scanning CSV files...")
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the specified directory.")
        
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        print("Loading chunks...")
        
        merged_chunks = []
        total_rows = 0
        
        for file_index, file in enumerate(csv_files):
            file_path = os.path.join(folder_path, file)
            print(f"Reading {file}...")
            
            try:
                chunk_count = 0
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
                    # Clean column names
                    chunk.columns = chunk.columns.str.strip()
                    
                    # Handle infinite values
                    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    # Drop rows with too many missing values
                    chunk = chunk.dropna(thresh=len(chunk.columns) * 0.7)
                    
                    if len(chunk) > 0:
                        merged_chunks.append(chunk)
                        total_rows += len(chunk)
                        chunk_count += 1
                    
                    if chunk_count >= max_chunks:
                        print(f"Reached {max_chunks} chunks for {file}, moving to next file.")
                        break
                        
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue
        
        if not merged_chunks:
            raise ValueError("No data could be loaded from any CSV files.")
        
        print("Concatenating all chunks...")
        merged_data = pd.concat(merged_chunks, ignore_index=True)
        del merged_chunks
        gc.collect()
        
        print(f"Merged dataset shape: {merged_data.shape}")
        print(f"Total rows loaded: {total_rows}")
        
        return merged_data

    def prepare_features(self, data, label_column='Label'):
        """Prepare features and labels for training."""
        print("Preparing features...")
        
        # Find label column
        possible_labels = ['Label', 'label', 'Class', 'class', 'Target', 'target']
        label_col = None
        
        for col in possible_labels:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            # Use last column as label
            label_col = data.columns[-1]
            print(f"Using last column '{label_col}' as label column")
        else:
            print(f"Using '{label_col}' as label column")
        
        # Separate features and labels
        X = data.drop(columns=[label_col])
        y = data[label_col]
        
        # Handle label encoding issues with proper character handling
        y = y.astype(str).str.strip()
        # Remove problematic characters and normalize
        y = y.str.encode('ascii', 'ignore').str.decode('ascii')
        y = y.str.replace(r'[^\w\s-]', '', regex=True)  # Keep only alphanumeric, spaces, and hyphens
        y = y.str.strip()
        
        # Handle empty labels
        y = y.replace('', 'Unknown')
        
        print(f"Unique labels found: {y.unique()[:10]}")  # Show first 10 unique labels
        
        # Keep only numeric features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            print(f"Dropping {len(non_numeric_columns)} non-numeric columns")
        
        X = X[numeric_columns]
        
        # Handle missing values
        print("Imputing missing values...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels with error handling
        try:
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"Successfully encoded {len(self.label_encoder.classes_)} unique labels")
        except Exception as e:
            print(f"Label encoding error: {e}")
            # Fallback: clean labels more aggressively
            y = y.str.replace(r'[^\w]', '_', regex=True)  # Replace all non-alphanumeric with underscore
            y_encoded = self.label_encoder.fit_transform(y)
            print("Used fallback label cleaning")
        
        print(f"Features shape: {X.shape}")
        print(f"Unique labels: {list(self.label_encoder.classes_)}")
        
        # Show class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("Class distribution:")
        for i, (label_idx, count) in enumerate(zip(unique, counts)):
            try:
                label_name = self.label_encoder.inverse_transform([label_idx])[0]
                print(f"  {label_name}: {count} samples")
            except Exception as e:
                print(f"  Label_{label_idx}: {count} samples")
        
        return X, y_encoded

    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest model with comprehensive evaluation."""
        print("Splitting data...")
        
        # Try stratified split first
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print("Using stratified split")
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print("Using random split (stratification not possible)")
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model training complete.")
        
        # Make predictions
        print("Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report with error handling
        print("\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        except Exception as e:
            print(f"Classification report error: {e}")
            print(classification_report(y_test, y_pred))
        
        # Generate visualizations
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance()
        
        return X_test_scaled, y_test, y_pred

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        print("Generating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        try:
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_
            )
        except Exception as e:
            print(f"Heatmap labeling error: {e}")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(r'C:\Users\Aizaz\Desktop\confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, top_n=15):
        """Plot top feature importances."""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print("Generating feature importance plot...")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig(r'C:\Users\Aizaz\Desktop\feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nTop {top_n} Important Features:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")

    def predict_threat(self, new_data):
        """Predict threats on new data."""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities

    def save_model(self, model_path="threat_detection_model.pkl"):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path="threat_detection_model.pkl"):
        """Load a saved model."""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from: {model_path}")

def main():
    """Main execution function."""
    # Initialize detector
    detector = CICIDSThreatDetector()
    
    # Set folder path containing CICIDS2017 CSV files
    folder_path = r'C:\Users\Aizaz\Downloads\MachineLearningCSV\MachineLearningCVE'
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        print("Please check the folder path and try again.")
        return
    
    try:
        print("Starting CICIDS2017 Threat Detection Training...")
        
        # Load and merge CSV files
        data = detector.load_and_merge_csvs(folder_path, max_chunks=5, chunk_size=10000)
        
        # Prepare features
        X, y = detector.prepare_features(data)
        
        # Train model
        X_test, y_test, y_pred = detector.train_model(X, y)
        
        # Save model to Desktop (change path as needed)
        model_path = r"C:\Users\Aizaz\Desktop\cicids_threat_model.pkl"
        detector.save_model(model_path)
        
        # Show sample predictions with error handling
        print("\nSample Predictions (First 5 test samples):")
        sample_predictions, sample_probs = detector.predict_threat(X_test[:5])
        for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probs)):
            max_prob = np.max(prob)
            try:
                print(f"Sample {i+1}: Predicted '{pred}' with confidence {max_prob:.4f}")
            except Exception as e:
                print(f"Sample {i+1}: Predicted 'Label_{pred}' with confidence {max_prob:.4f}")
        
        print("\nTraining completed successfully!")
        print("Model saved to Desktop: 'cicids_threat_model.pkl'")
        print("Plots saved to Desktop: 'confusion_matrix.png' and 'feature_importance.png'")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Trying with smaller chunks...")
        
        try:
            # Retry with smaller parameters
            data = detector.load_and_merge_csvs(folder_path, max_chunks=3, chunk_size=5000)
            X, y = detector.prepare_features(data)
            X_test, y_test, y_pred = detector.train_model(X, y)
            detector.save_model(r"C:\Users\Aizaz\Desktop\cicids_threat_model.pkl")
            print("Training completed with smaller chunks!")
            
        except Exception as e2:
            print(f"Still having issues: {str(e2)}")
            print("Consider:")
            print("1. Reducing chunk_size further (e.g., 2000)")
            print("2. Reducing max_chunks (e.g., 2)")
            print("3. Checking available system memory")

if __name__ == "__main__":
    main()
