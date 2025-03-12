# ML.NET Consolidation Prediction Engine

## Overview

This machine learning solution predicts consolidation patterns from raw data, determining if a message should create a parent record, how many child records to create, and what data these records should contain. It uses ML.NET (C#) or scikit-learn (Python) to learn from historical consolidation patterns.

## Table of Contents

- [Data Relationships](#data-relationships)
- [Features](#features)
- [Implementation Options](#implementation-options)
  - [.NET 8 Implementation](#net-8-implementation)
  - [Python Implementation](#python-implementation)
- [Required Training Data](#required-training-data)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Benefits](#benefits)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Integration Examples](#integration-examples)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Data Relationships

The system handles these entity relationships:

- **Raw data** contains `messageid` and `conversionid`
- `messageid` is a unique key, with `conversionid` having a one-to-many relationship to `messageid`
- Each `conversionid` may create a consolidated parent based on business rules
- Each parent should have 1 or more children

## Features

- **Predictive Decision-Making**: Determines whether to create parent records
- **Child Count Prediction**: Estimates the number of child records to create
- **Content Generation**: Predicts actual values for parent and child records
- **Pattern-Based Learning**: Uses historical data to identify consolidation patterns
- **Batch Processing**: Efficiently handles multiple records
- **Template-Based Generation**: Creates realistic data based on historical patterns
- **Fallback Mechanisms**: Provides sensible defaults when no patterns match

## Implementation Options

### .NET 8 Implementation

The .NET implementation uses ML.NET's FastTree algorithms for classification and regression, along with template-based content generation for predicting record data.

### Python Implementation

The Python implementation uses scikit-learn's RandomForest algorithms for classification and regression, with similar template-based generation for content prediction.

## Required Training Data

Your training data needs these columns:

1. `MessageId` - Unique identifier for the message
2. `ConversionId` - Identifier for the conversion
3. Feature columns - Any attributes influencing consolidation rules:
   - `AmountValue` - Transaction amount
   - `TransactionAge` - Age of the transaction
   - `Category` - Category code
   - `Source` - Source system
   - `Description` - Text description
   - *Other domain-specific features*
4. `HasParent` - Boolean indicating if a parent was created
5. `ChildCount` - Number of child records created
6. `ParentData` - JSON string of the parent record (if any)
7. `ChildrenData` - JSON string of the child records (if any)

### Example CSV Format:

MessageId,ConversionId,AmountValue,Category,Source,Description,HasParent,ChildCount,ParentData,ChildrenData
MSG001,CONV1,1200.50,TypeA,System1,"Payment received",true,2,"{\"parent_id\":\"P123\",\"name\":\"Invoice Payment\"}","[{\"child_id\":\"C001\",\"amount\":700.50}]"
MSG002,CONV2,800.25,TypeB,System2,"Transfer initiated",true,1,"{\"parent_id\":\"P124\"}","[{\"child_id\":\"C003\"}]"
MSG003,CONV1,450.75,TypeA,System1,"Additional payment",false,0,"",""

## How It Works

### .NET 8 Implementation

```csharp
// Train the models
var engine = new ConsolidationEngine();
engine.TrainModels("training_data.csv");
engine.SaveModels("parent_model.zip", "child_model.zip", "templates.json");

// Make predictions
var input = new RawDataInput {
    MessageId = "MSG12345",
    ConversionId = "CONV789",
    AmountValue = 1250.50f,
    Category = "TypeA",
    Source = "System1",
    Description = "Sample transaction"
};

var prediction = engine.Predict(input);

// Access prediction results
if (prediction.ShouldCreateParent) {
    Console.WriteLine($"Parent: {prediction.PredictedParent.Name}");
    Console.WriteLine($"Children: {prediction.PredictedChildCount}");
    
    foreach (var child in prediction.PredictedChildren) {
        Console.WriteLine($"  Child: {child.ChildId}, Amount: {child.Amount}");
    }
}

# Train the models
engine = ConsolidationEngine()
engine.train_models("training_data.csv")
engine.save_models("parent_model.pkl", "child_model.pkl", "templates.json")

# Make predictions
input_data = {
    'MessageId': 'MSG12345',
    'ConversionId': 'CONV789',
    'AmountValue': 1250.50,
    'Category': 'TypeA',
    'Source': 'System1',
    'Description': 'Sample transaction'
}

prediction = engine.predict(input_data)

# Access prediction results
if prediction.should_create_parent:
    print(f"Parent: {prediction.predicted_parent.name}")
    print(f"Children: {prediction.predicted_child_count}")
    
    for child in prediction.predicted_children:
        print(f"  Child: {child.child_id}, Amount: {child.amount:.2f}")

### Training Phase:

1. **Data Loading**:
   - Loads historical consolidation data from CSV
   - Parses feature data and labels
   - Extracts parent and child record templates

2. **Model Training**:
   - Trains a binary classifier to predict parent creation
   - Trains a regression model to predict child count
   - Evaluates model performance with metrics

3. **Template Extraction**:
   - Parses JSON data for successful consolidations
   - Categorizes templates by patterns (e.g., Category + Source)
   - Stores templates for content generation

### Prediction Phase:

1. **Decision Making**:
   - Predicts whether to create a parent record
   - If yes, predicts how many children to create

2. **Content Generation**:
   - Selects appropriate templates based on input features
   - Customizes templates with specific values from input
   - Generates unique identifiers for new records
   - Distributes amounts logically among child records

3. **Result Assembly**:
   - Combines predictions into a complete result
   - Includes both decision outcomes and generated content

## Model Architecture

### Parent Prediction Model:
- **Type**: Binary Classification
- **Algorithm**: Fast Tree / Random Forest
- **Features**:
  - Numerical features (normalized)
  - Categorical features (one-hot encoded)
  - Text descriptions (feature hashed)

### Child Count Model:
- **Type**: Regression
- **Algorithm**: Fast Tree / Random Forest
- **Features**: Same as parent model
- **Target**: Number of children to create

### Content Generation:
- **Method**: Template-based generation
- **Template Selection**: Pattern matching on features
- **Customization**: Feature-based value substitution
- **Fallback**: Default templates when no match found

## Benefits

- **Learn vs. Code**: Learns patterns from data instead of hard-coding business rules
- **Adaptable**: As consolidation patterns evolve, simply retrain with new data
- **Content Prediction**: Goes beyond yes/no decisions to predict record contents
- **Comprehensive**: Handles the entire consolidation workflow
- **Probabilistic**: Provides confidence scores for predictions
- **Scalable**: Handles batch processing for high-volume scenarios
- **Templated**: Creates realistic data based on historical patterns

## Requirements

### .NET Implementation
- .NET 8.0
- ML.NET 2.0+
- Newtonsoft.Json

### Python Implementation
- Python 3.8+
- scikit-learn
- pandas
- numpy
- joblib

## Getting Started

1. **Prepare Training Data**:
   - Collect historical message data and consolidation outcomes
   - Format data according to the required schema
   - Ensure parent and child data is properly JSON-encoded

2. **Train the Models**
   - Use the ConsolidationEngine to train models with your data
   - Save the trained models for later use

3. **Deploy for Prediction**:
   - Load the trained models in your application
   - Pass raw message data to get predictions

4. **Integrate with Your Application**:
   - Convert your raw message data to the input format
   - Call the prediction engine
   - Process the results to create actual records

## Integration Examples

### ASP.NET Web API Integration

Create a Web API controller that exposes endpoints for prediction, both for single and batch processing.

```csharp
[ApiController]
[Route("api/[controller]")]
public class ConsolidationController : ControllerBase
{
    private readonly ConsolidationEngine _engine;
    
    public ConsolidationController(ConsolidationEngine engine)
    {
        _engine = engine;
    }
    
    [HttpPost("predict")]
    public ActionResult<ConsolidationPrediction> Predict(RawDataInput input)
    {
        var prediction = _engine.Predict(input);
        return Ok(prediction);
    }
    
    [HttpPost("batch")]
    public ActionResult<List<ConsolidationPrediction>> PredictBatch(List<RawDataInput> inputs)
    {
        var predictions = _engine.PredictBatch(inputs);
        return Ok(predictions);
    }
}
```

### Python Flask API Integration

Create a Flask API that provides prediction endpoints for both single messages and batch processing.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
engine = ConsolidationEngine()
engine.load_models("parent_model.pkl", "child_model.pkl", "templates.json")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction = engine.predict(input_data)
    return jsonify(prediction.to_dict())

@app.route('/batch', methods=['POST'])
def predict_batch():
    inputs = request.json
    predictions = engine.predict_batch(inputs)
    return jsonify([p.to_dict() for p in predictions])

if __name__ == '__main__':
    app.run(debug=True)
```

## Performance Considerations

- **Batch Processing**: For high volume scenarios, use batch processing
- **Caching**: Consider caching templates for faster content generation
- **Scaling**: ML models can be deployed to multiple instances
- **Monitoring**: Track prediction accuracy over time
- **Retraining**: Periodically retrain with new data to maintain accuracy

## Troubleshooting

### Common Issues:

1. **Low Prediction Accuracy**:
   - Ensure training data has sufficient examples
   - Add more relevant features
   - Try different ML algorithms

2. **Content Generation Issues**:
   - Verify JSON format in training data
   - Add more diverse templates
   - Check pattern matching logic

3. **Performance Problems**:
   - Use batch processing for high volume
   - Optimize feature preprocessing
   - Consider hardware acceleration

## FAQ

**Q: How much training data is needed?**  
A: For good results, aim for at least 1,000 examples with a balanced distribution of outcomes.

**Q: How often should models be retrained?**  
A: Retrain when business rules change or prediction accuracy drops. Typically every few months.

**Q: Can this handle complex business rules?**  
A: Yes, by encoding the rules into features. The ML model will learn the patterns from examples.

**Q: Is it possible to explain why a prediction was made?**  
A: ML.NET provides feature importance metrics that can help explain which factors influenced a decision.

**Q: How to handle new categories not seen in training?**  
A: The system uses one-hot encoding with "unknown" handling. It will make predictions based on other features.

---

*For more information or support, please contact the development team.*