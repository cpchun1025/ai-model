import os
import pandas as pd
import numpy as np
import json
import uuid
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib
from typing import List, Dict, Any, Tuple, Optional

# Parent record model
class ParentRecord:
    def __init__(self, parent_id: str = None, name: str = None, total_amount: float = 0,
                 status: str = "New", category: str = None, created_date: datetime = None):
        self.parent_id = parent_id or f"P-{uuid.uuid4().hex}"
        self.name = name
        self.total_amount = total_amount
        self.status = status
        self.category = category
        self.created_date = created_date or datetime.now()
    
    def to_dict(self):
        return {
            "parent_id": self.parent_id,
            "name": self.name,
            "total_amount": self.total_amount,
            "status": self.status,
            "category": self.category,
            "created_date": self.created_date.isoformat() if isinstance(self.created_date, datetime) else self.created_date
        }
    
    @classmethod
    def from_dict(cls, data):
        if isinstance(data, dict):
            return cls(
                parent_id=data.get("parent_id"),
                name=data.get("name"),
                total_amount=data.get("total_amount", 0),
                status=data.get("status", "New"),
                category=data.get("category"),
                created_date=data.get("created_date")
            )
        return None

# Child record model
class ChildRecord:
    def __init__(self, child_id: str = None, parent_id: str = None, amount: float = 0,
                 type: str = None, description: str = None, transaction_date: datetime = None):
        self.child_id = child_id or f"C-{uuid.uuid4().hex}"
        self.parent_id = parent_id
        self.amount = amount
        self.type = type or "Standard"
        self.description = description
        self.transaction_date = transaction_date or datetime.now()
    
    def to_dict(self):
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "amount": self.amount,
            "type": self.type,
            "description": self.description,
            "transaction_date": self.transaction_date.isoformat() if isinstance(self.transaction_date, datetime) else self.transaction_date
        }
    
    @classmethod
    def from_dict(cls, data):
        if isinstance(data, dict):
            return cls(
                child_id=data.get("child_id"),
                parent_id=data.get("parent_id"),
                amount=data.get("amount", 0),
                type=data.get("type", "Standard"),
                description=data.get("description"),
                transaction_date=data.get("transaction_date")
            )
        return None

# Consolidation prediction result
class ConsolidationPrediction:
    def __init__(self, message_id: str, conversion_id: str):
        self.message_id = message_id
        self.conversion_id = conversion_id
        self.should_create_parent = False
        self.parent_confidence = 0.0
        self.predicted_child_count = 0
        self.predicted_parent = None
        self.predicted_children = []
    
    def to_dict(self):
        return {
            "message_id": self.message_id,
            "conversion_id": self.conversion_id,
            "should_create_parent": self.should_create_parent,
            "parent_confidence": self.parent_confidence,
            "predicted_child_count": self.predicted_child_count,
            "predicted_parent": self.predicted_parent.to_dict() if self.predicted_parent else None,
            "predicted_children": [child.to_dict() for child in self.predicted_children]
        }

class ConsolidationEngine:
    def __init__(self):
        self.parent_model = None
        self.child_model = None
        self.preprocessor = None
        self.parent_templates = {}  # Pattern key -> list of ParentRecord templates
        self.child_templates = {}   # Pattern key -> list of ChildRecord templates
    
    def train_models(self, training_data_path: str) -> None:
        """Train both models using data from the provided path"""
        print("Loading training data...")
        
        # Load training data
        data = pd.read_csv(training_data_path)
        
        # Separate features and targets
        X = data.drop(['HasParent', 'ChildCount', 'ParentData', 'ChildrenData'], axis=1, errors='ignore')
        y_parent = data['HasParent']
        y_child = data['ChildCount']
        
        # Split numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns from features
        id_columns = ['MessageId', 'ConversionId']
        numeric_features = [col for col in numeric_features if col not in id_columns]
        categorical_features = [col for col in categorical_features if col not in id_columns]
        
        # Create feature preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Split into training and test sets
        X_train, X_test, y_parent_train, y_parent_test, y_child_train, y_child_test = train_test_split(
            X, y_parent, y_child, test_size=0.2, random_state=42
        )
        
        # Train parent prediction model
        print("Training parent prediction model...")
        parent_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42))
        ])
        
        parent_pipeline.fit(X_train, y_parent_train)
        
        # Evaluate parent model
        y_parent_pred = parent_pipeline.predict(X_test)
        parent_accuracy = accuracy_score(y_parent_test, y_parent_pred)
        parent_f1 = f1_score(y_parent_test, y_parent_pred)
        
        print(f"Parent model accuracy: {parent_accuracy:.2%}")
        print(f"Parent model F1 score: {parent_f1:.2%}")
        
        # Train child count prediction model
        print("Training child count prediction model...")
        child_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
        ])
        
        # Only train on rows that have a parent
        parent_mask = y_parent_train == True
        if sum(parent_mask) > 0:  # Make sure we have examples with parents
            child_pipeline.fit(X_train[parent_mask], y_child_train[parent_mask])
            
            # Evaluate child model on test examples that have parents
            test_parent_mask = y_parent_test == True
            if sum(test_parent_mask) > 0:
                y_child_pred = child_pipeline.predict(X_test[test_parent_mask])
                child_r2 = r2_score(y_child_test[test_parent_mask], y_child_pred)
                child_rmse = np.sqrt(mean_squared_error(y_child_test[test_parent_mask], y_child_pred))
                
                print(f"Child model R²: {child_r2:.2f}")
                print(f"Child model RMSE: {child_rmse:.2f}")
        
        # Save the trained models
        self.parent_model = parent_pipeline
        self.child_model = child_pipeline
        self.preprocessor = preprocessor
        
        # Extract content templates
        self.extract_content_templates(training_data_path)
        
    def extract_content_templates(self, training_data_path: str) -> None:
        """Extract content templates from the training data"""
        print("Extracting content templates from training data...")
        
        try:
            # Load the training data
            data = pd.read_csv(training_data_path)
            
            # Only consider rows that have a parent
            parent_data = data[data['HasParent'] == True]
            
            for _, row in parent_data.iterrows():
                # Create a pattern key based on category and source
                category = str(row.get('Category', ''))
                source = str(row.get('Source', ''))
                pattern_key = f"{category}_{source}"
                
                # Extract parent data
                try:
                    parent_json = row.get('ParentData', '{}')
                    if isinstance(parent_json, str):
                        parent_data = json.loads(parent_json)
                        parent_record = ParentRecord.from_dict(parent_data)
                        
                        if parent_record:
                            # Store parent template
                            if pattern_key not in self.parent_templates:
                                self.parent_templates[pattern_key] = []
                            self.parent_templates[pattern_key].append(parent_record)
                            
                            # Extract children data
                            children_json = row.get('ChildrenData', '[]')
                            if isinstance(children_json, str):
                                children_data = json.loads(children_json)
                                
                                child_records = []
                                if isinstance(children_data, list):
                                    for child_data in children_data:
                                        child_record = ChildRecord.from_dict(child_data)
                                        if child_record:
                                            child_records.append(child_record)
                                
                                # Store child templates
                                if pattern_key not in self.child_templates:
                                    self.child_templates[pattern_key] = []
                                self.child_templates[pattern_key].extend(child_records)
                except Exception as e:
                    print(f"Error processing template for row: {e}")
                    continue
            
            print(f"Extracted {len(self.parent_templates)} parent template patterns")
            for key, templates in self.parent_templates.items():
                print(f"  Pattern '{key}': {len(templates)} templates")
                
            print(f"Extracted {len(self.child_templates)} child template patterns")
            for key, templates in self.child_templates.items():
                print(f"  Pattern '{key}': {len(templates)} templates")
                
        except Exception as e:
            print(f"Error extracting templates: {e}")
    
    def save_models(self, parent_model_path: str, child_model_path: str, templates_path: str) -> None:
        """Save the trained models and templates to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(parent_model_path), exist_ok=True)
        
        # Save ML models
        joblib.dump(self.parent_model, parent_model_path)
        joblib.dump(self.child_model, child_model_path)
        
        # Save templates
        templates_data = {
            "parent_templates": {
                k: [template.to_dict() for template in v] 
                for k, v in self.parent_templates.items()
            },
            "child_templates": {
                k: [template.to_dict() for template in v] 
                for k, v in self.child_templates.items()
            }
        }
        
        with open(templates_path, 'w') as f:
            json.dump(templates_data, f, indent=2)
        
        print(f"Models and templates saved to {os.path.dirname(parent_model_path)}")
        
    def load_models(self, parent_model_path: str, child_model_path: str, templates_path: str) -> None:
        """Load trained models and templates from disk"""
        # Load ML models
        self.parent_model = joblib.load(parent_model_path)
        self.child_model = joblib.load(child_model_path)
        
        # Load templates
        if os.path.exists(templates_path):
            with open(templates_path, 'r') as f:
                templates_data = json.load(f)
            
            # Load parent templates
            self.parent_templates = {}
            for pattern_key, templates in templates_data.get("parent_templates", {}).items():
                self.parent_templates[pattern_key] = [ParentRecord.from_dict(t) for t in templates]
            
            # Load child templates
            self.child_templates = {}
            for pattern_key, templates in templates_data.get("child_templates", {}).items():
                self.child_templates[pattern_key] = [ChildRecord.from_dict(t) for t in templates]
        
        print("Models and templates loaded successfully")
        
    def predict(self, raw_input: Dict[str, Any]) -> ConsolidationPrediction:
        """Make a prediction for a single input record"""
        if self.parent_model is None or self.child_model is None:
            raise ValueError("Models must be trained or loaded before making predictions")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([raw_input])
        
        # Create pattern key for this input
        category = str(raw_input.get('Category', ''))
        source = str(raw_input.get('Source', ''))
        pattern_key = f"{category}_{source}"
        
        # Make parent prediction
        parent_prob = self.parent_model.predict_proba(input_df)[0, 1]
        should_create_parent = self.parent_model.predict(input_df)[0]
        
        # Initialize result
        result = ConsolidationPrediction(
            message_id=raw_input.get('MessageId', ''),
            conversion_id=raw_input.get('ConversionId', '')
        )
        result.should_create_parent = bool(should_create_parent)
        result.parent_confidence = float(parent_prob)
        
        # If a parent should be created, predict child count and generate content
        if should_create_parent:
            # Predict child count
            child_count_raw = self.child_model.predict(input_df)[0]
            child_count = max(1, round(child_count_raw))  # At least 1 child
            result.predicted_child_count = int(child_count)
            
            # Generate parent content
            result.predicted_parent = self.generate_parent_content(raw_input, pattern_key)
            
            # Generate child content
            result.predicted_children = self.generate_child_content(
                raw_input, pattern_key, child_count, result.predicted_parent.parent_id
            )
            
        return result
    
    def generate_parent_content(self, raw_input: Dict[str, Any], pattern_key: str) -> ParentRecord:
        """Generate parent content based on templates or defaults if no templates exist"""
        # Try to find a template matching the pattern
        if pattern_key in self.parent_templates and self.parent_templates[pattern_key]:
            # Select a random template as the base
            template = random.choice(self.parent_templates[pattern_key])
            
            # Clone and customize the template
            parent = ParentRecord(
                name=f"{template.name or 'Parent'} {uuid.uuid4().hex[:6]}",
                total_amount=float(raw_input.get('AmountValue', 0)),
                status=template.status or "New",
                category=raw_input.get('Category'),
                created_date=datetime.now()
            )
            
            return parent
        
        # Fallback if no matching template
        return ParentRecord(
            name=f"Auto-generated from {raw_input.get('ConversionId', 'unknown')}",
            total_amount=float(raw_input.get('AmountValue', 0)),
            status="New",
            category=raw_input.get('Category'),
            created_date=datetime.now()
        )
    
    def generate_child_content(self, raw_input: Dict[str, Any], pattern_key: str, 
                              child_count: int, parent_id: str) -> List[ChildRecord]:
        """Generate child content based on templates or defaults if no templates exist"""
        children = []
        
        # Generate random distribution of the total amount
        total_amount = float(raw_input.get('AmountValue', 0))
        amounts = self.generate_random_distribution(total_amount, child_count)
        
        # Try to find templates matching the pattern
        if pattern_key in self.child_templates and self.child_templates[pattern_key]:
            templates = self.child_templates[pattern_key]
            
            for i in range(child_count):
                # Select a random template
                template = random.choice(templates)
                
                # Create child based on template
                child = ChildRecord(
                    parent_id=parent_id,
                    amount=amounts[i],
                    type=template.type or "Standard",
                    description=raw_input.get('Description') or template.description or f"Child {i+1}",
                    transaction_date=datetime.now() - timedelta(days=random.randint(0, 5))
                )
                
                children.append(child)
        else:
            # Fallback if no matching templates
            for i in range(child_count):
                child = ChildRecord(
                    parent_id=parent_id,
                    amount=amounts[i],
                    type="Standard",
                    description=raw_input.get('Description') or f"Auto-generated child {i+1}",
                    transaction_date=datetime.now()
                )
                
                children.append(child)
        
        return children
    
    def generate_random_distribution(self, total: float, parts: int) -> List[float]:
        """Generate a random distribution of values that sum to the total"""
        if parts <= 0:
            return []
        if parts == 1:
            return [total]
        
        # Generate random values
        distribution = [random.random() for _ in range(parts)]
        total_random = sum(distribution)
        
        # Normalize to the requested total
        return [(value / total_random) * total for value in distribution]
    
    def predict_batch(self, raw_inputs: List[Dict[str, Any]]) -> List[ConsolidationPrediction]:
        """Make predictions for a batch of input records"""
        return [self.predict(input_record) for input_record in raw_inputs]


# Example usage
if __name__ == "__main__":
    # Initialize the engine
    engine = ConsolidationEngine()
    
    # Define paths
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)
    
    training_data_path = os.path.join(data_path, "training_data.csv")
    parent_model_path = os.path.join(data_path, "parent_model.pkl")
    child_model_path = os.path.join(data_path, "child_model.pkl")
    templates_path = os.path.join(data_path, "content_templates.json")
    
    try:
        # Check if training data exists
        if os.path.exists(training_data_path):
            # Either train new models
            engine.train_models(training_data_path)
            engine.save_models(parent_model_path, child_model_path, templates_path)
        
            # Or load existing models if they exist
            # engine.load_models(parent_model_path, child_model_path, templates_path)
            
            # Make a prediction on a single record
            sample_input = {
                'MessageId': 'MSG12345',
                'ConversionId': 'CONV789',
                'AmountValue': 1250.50,
                'TransactionAge': 3.5,
                'Category': 'TypeA',
                'Source': 'System1',
                'Description': 'Sample transaction'
            }
            
            prediction = engine.predict(sample_input)
            
            print(f"Message ID: {prediction.message_id}")
            print(f"Conversion ID: {prediction.conversion_id}")
            print(f"Should create parent: {prediction.should_create_parent} (Confidence: {prediction.parent_confidence:.2%})")
            
            if prediction.should_create_parent:
                print(f"Predicted child count: {prediction.predicted_child_count}")
                
                # Display parent details
                parent = prediction.predicted_parent
                print(f"\nPredicted Parent:")
                print(f"  ID: {parent.parent_id}")
                print(f"  Name: {parent.name}")
                print(f"  Total Amount: ${parent.total_amount:.2f}")
                print(f"  Status: {parent.status}")
                print(f"  Category: {parent.category}")
                
                # Display children details
                print(f"\nPredicted Children:")
                for i, child in enumerate(prediction.predicted_children):
                    print(f"  Child {i+1}:")
                    print(f"    ID: {child.child_id}")
                    print(f"    Amount: ${child.amount:.2f}")
                    print(f"    Type: {child.type}")
                    print(f"    Description: {child.description}")
            
            # Process a batch of raw data
            print("\nProcessing a batch of records...")
            batch_inputs = [
                {'MessageId': 'MSG1001', 'ConversionId': 'CONV101', 'AmountValue': 500, 'TransactionAge': 1.2, 
                 'Category': 'TypeB', 'Source': 'System2', 'Description': 'First transaction'},
                {'MessageId': 'MSG1002', 'ConversionId': 'CONV102', 'AmountValue': 1800, 'TransactionAge': 5.7, 
                 'Category': 'TypeA', 'Source': 'System1', 'Description': 'Second transaction'},
                {'MessageId': 'MSG1003', 'ConversionId': 'CONV101', 'AmountValue': 350, 'TransactionAge': 2.1, 
                 'Category': 'TypeC', 'Source': 'System3', 'Description': 'Third transaction'}
            ]
            
            batch_results = engine.predict_batch(batch_inputs)
            
            for result in batch_results:
                print(f"Message: {result.message_id}, Create Parent: {result.should_create_parent}, Child Count: {result.predicted_child_count}")
                
                if result.should_create_parent:
                    # Show summary of parent and children
                    parent = result.predicted_parent
                    print(f"  Parent: {parent.name} (${parent.total_amount:.2f})")
                    
                    # Summarize children
                    child_types = {}
                    for child in result.predicted_children:
                        child_type = child.type
                        if child_type in child_types:
                            child_types[child_type] += 1
                        else:
                            child_types[child_type] = 1
                    
                    print(f"  Children: {', '.join([f'{count} {type}' for type, count in child_types.items()])}")
        else:
            print(f"Training data not found at {training_data_path}")
            print("Please create a training dataset with the required columns.")
            
    except Exception as e:
        print(f"Error: {e}")