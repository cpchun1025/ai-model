using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

namespace ConsolidationPredictionEngine
{
    // Input data class from your raw data table
    public class RawDataInput
    {
        [LoadColumn(0)]
        public string MessageId { get; set; }
        
        [LoadColumn(1)]
        public string ConversionId { get; set; }
        
        // Add features that influence your business rules
        [LoadColumn(2)]
        public float AmountValue { get; set; }
        
        [LoadColumn(3)]
        public float TransactionAge { get; set; }
        
        [LoadColumn(4)]
        public string Category { get; set; }
        
        [LoadColumn(5)]
        public string Source { get; set; }
        
        [LoadColumn(6)]
        public string Description { get; set; }
        
        // Add more columns as needed
    }

    // Model for a parent record
    public class ParentRecord
    {
        public string ParentId { get; set; }
        public string Name { get; set; }
        public float TotalAmount { get; set; }
        public string Status { get; set; }
        public string Category { get; set; }
        public DateTime CreatedDate { get; set; }
        // Add more properties that represent your parent record
    }
    
    // Model for a child record
    public class ChildRecord
    {
        public string ChildId { get; set; }
        public string ParentId { get; set; }
        public float Amount { get; set; }
        public string Type { get; set; }
        public string Description { get; set; }
        public DateTime TransactionDate { get; set; }
        // Add more properties that represent your child record
    }

    // Prediction output including content
    public class ConsolidationPrediction
    {
        // Original IDs for reference
        public string MessageId { get; set; }
        public string ConversionId { get; set; }
        
        // Basic prediction results
        public bool ShouldCreateParent { get; set; }
        public float ParentConfidence { get; set; }
        public int PredictedChildCount { get; set; }
        
        // Predicted content
        public ParentRecord PredictedParent { get; set; }
        public List<ChildRecord> PredictedChildren { get; set; }
    }

    // Class for training data with parent and child content
    public class TrainingData
    {
        [LoadColumn(0)]
        public string MessageId { get; set; }
        
        [LoadColumn(1)]
        public string ConversionId { get; set; }
        
        // Features
        [LoadColumn(2)]
        public float AmountValue { get; set; }
        
        [LoadColumn(3)]
        public float TransactionAge { get; set; }
        
        [LoadColumn(4)]
        public string Category { get; set; }
        
        [LoadColumn(5)]
        public string Source { get; set; }
        
        [LoadColumn(6)]
        public string Description { get; set; }
        
        // Labels for basic prediction
        [LoadColumn(7)]
        [ColumnName("Label")]
        public bool HasParent { get; set; }
        
        [LoadColumn(8)]
        [ColumnName("ChildCount")]
        public float ChildCount { get; set; }
        
        // Parent and child content in JSON format for content prediction
        [LoadColumn(9)]
        public string ParentData { get; set; }
        
        [LoadColumn(10)]
        public string ChildrenData { get; set; }
    }

    // Classes for internal predictions
    internal class ParentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool ShouldCreateParent { get; set; }
        
        [ColumnName("Probability")]
        public float Probability { get; set; }
    }
    
    internal class ChildCountPrediction
    {
        [ColumnName("Score")]
        public float PredictedChildCount { get; set; }
    }

    class ConsolidationEngine
    {
        private readonly MLContext _mlContext;
        private ITransformer _parentModel;
        private ITransformer _childModel;
        private PredictionEngine<RawDataInput, ParentPrediction> _parentPredictionEngine;
        private PredictionEngine<RawDataInput, ChildCountPrediction> _childPredictionEngine;
        
        // Dictionary to store templates based on patterns
        private Dictionary<string, List<ParentRecord>> _parentTemplates;
        private Dictionary<string, List<ChildRecord>> _childTemplates;
        
        public ConsolidationEngine()
        {
            _mlContext = new MLContext(seed: 0);
            _parentTemplates = new Dictionary<string, List<ParentRecord>>();
            _childTemplates = new Dictionary<string, List<ChildRecord>>();
        }
        
        public void TrainModels(string trainingDataPath)
        {
            Console.WriteLine("Loading training data...");
            
            // Load data
            var trainingData = _mlContext.Data.LoadFromTextFile<TrainingData>(
                trainingDataPath, hasHeader: true, separatorChar: ',');
                
            // Split for training and evaluation
            var dataSplit = _mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);
            var trainSet = dataSplit.TrainSet;
            var testSet = dataSplit.TestSet;
            
            TrainParentModel(trainSet, testSet);
            TrainChildModel(trainSet, testSet);
            
            // Extract content templates from training data
            ExtractContentTemplates(trainingDataPath);
        }
        
        private void TrainParentModel(IDataView trainData, IDataView testData)
        {
            Console.WriteLine("Training parent prediction model...");
            
            // Create feature processing pipeline
            var dataPipeline = _mlContext.Transforms.Categorical.OneHotEncoding("CategoryEncoded", "Category")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("SourceEncoded", "Source"))
                .Append(_mlContext.Transforms.Text.FeaturizeText("DescriptionFeatures", "Description"))
                .Append(_mlContext.Transforms.Concatenate("Features", 
                    "AmountValue", "TransactionAge", "CategoryEncoded", "SourceEncoded", "DescriptionFeatures"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"));
            
            // Add the algorithm
            var trainingPipeline = dataPipeline
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    numberOfLeaves: 20, 
                    numberOfTrees: 100, 
                    minimumExampleCountPerLeaf: 10));
            
            // Train model
            _parentModel = trainingPipeline.Fit(trainData);
            
            // Evaluate model
            var predictions = _parentModel.Transform(testData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);
            
            Console.WriteLine($"Parent model accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Parent model F1 score: {metrics.F1Score:P2}");
            
            // Create prediction engine
            _parentPredictionEngine = _mlContext.Model.CreatePredictionEngine<RawDataInput, ParentPrediction>(_parentModel);
        }
        
        private void TrainChildModel(IDataView trainData, IDataView testData)
        {
            Console.WriteLine("Training child count prediction model...");
            
            // Create feature processing pipeline
            var dataPipeline = _mlContext.Transforms.Categorical.OneHotEncoding("CategoryEncoded", "Category")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("SourceEncoded", "Source"))
                .Append(_mlContext.Transforms.Text.FeaturizeText("DescriptionFeatures", "Description"))
                .Append(_mlContext.Transforms.Concatenate("Features", 
                    "AmountValue", "TransactionAge", "CategoryEncoded", "SourceEncoded", "DescriptionFeatures"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"));
            
            // Add the algorithm
            var trainingPipeline = dataPipeline
                .Append(_mlContext.Regression.Trainers.FastTree(
                    numberOfLeaves: 20, 
                    numberOfTrees: 100, 
                    labelColumnName: "ChildCount"));
            
            // Train model
            _childModel = trainingPipeline.Fit(trainData);
            
            // Evaluate model
            var predictions = _childModel.Transform(testData);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "ChildCount");
            
            Console.WriteLine($"Child count model R²: {metrics.RSquared:F2}");
            Console.WriteLine($"Child count model RMSE: {metrics.RootMeanSquaredError:F2}");
            
            // Create prediction engine
            _childPredictionEngine = _mlContext.Model.CreatePredictionEngine<RawDataInput, ChildCountPrediction>(_childModel);
        }
        
        private void ExtractContentTemplates(string trainingDataPath)
        {
            Console.WriteLine("Extracting content templates from training data...");
            
            // Read the training data as CSV
            var lines = File.ReadAllLines(trainingDataPath).Skip(1); // Skip header
            
            foreach (var line in lines)
            {
                var fields = line.Split(',');
                if (fields.Length < 11) continue; // Need at least all required fields
                
                // Extract key fields
                string category = fields[4];
                string source = fields[5];
                bool hasParent = bool.Parse(fields[7]);
                
                if (!hasParent) continue; // Skip records without parents
                
                // Create a pattern key for categorizing templates
                string patternKey = $"{category}_{source}";
                
                // Extract parent data
                string parentJson = fields[9].Replace("\"\"", "\""); // Fix escaped quotes
                if (parentJson.StartsWith("\"") && parentJson.EndsWith("\""))
                {
                    parentJson = parentJson.Substring(1, parentJson.Length - 2);
                }
                
                try
                {
                    var parentRecord = JsonConvert.DeserializeObject<ParentRecord>(parentJson);
                    
                    // Store parent template
                    if (!_parentTemplates.ContainsKey(patternKey))
                    {
                        _parentTemplates[patternKey] = new List<ParentRecord>();
                    }
                    _parentTemplates[patternKey].Add(parentRecord);
                    
                    // Extract children data
                    string childrenJson = fields[10].Replace("\"\"", "\""); // Fix escaped quotes
                    if (childrenJson.StartsWith("\"") && childrenJson.EndsWith("\""))
                    {
                        childrenJson = childrenJson.Substring(1, childrenJson.Length - 2);
                    }
                    
                    var childRecords = JsonConvert.DeserializeObject<List<ChildRecord>>(childrenJson);
                    
                    // Store child templates
                    if (!_childTemplates.ContainsKey(patternKey))
                    {
                        _childTemplates[patternKey] = new List<ChildRecord>();
                    }
                    _childTemplates[patternKey].AddRange(childRecords);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing JSON template: {ex.Message}");
                }
            }
            
            Console.WriteLine($"Extracted {_parentTemplates.Count} parent template patterns and {_childTemplates.Count} child template patterns");
        }
        
        public void SaveModels(string parentModelPath, string childModelPath, string templatesPath)
        {
            // Save ML models
            _mlContext.Model.Save(_parentModel, null, parentModelPath);
            _mlContext.Model.Save(_childModel, null, childModelPath);
            
            // Save templates
            var templates = new
            {
                ParentTemplates = _parentTemplates,
                ChildTemplates = _childTemplates
            };
            
            File.WriteAllText(templatesPath, JsonConvert.SerializeObject(templates, Formatting.Indented));
            
            Console.WriteLine($"Models and templates saved successfully");
        }
        
        public void LoadModels(string parentModelPath, string childModelPath, string templatesPath)
        {
            // Load ML models
            _parentModel = _mlContext.Model.Load(parentModelPath, out var _);
            _childModel = _mlContext.Model.Load(childModelPath, out var _);
            
            _parentPredictionEngine = _mlContext.Model.CreatePredictionEngine<RawDataInput, ParentPrediction>(_parentModel);
            _childPredictionEngine = _mlContext.Model.CreatePredictionEngine<RawDataInput, ChildCountPrediction>(_childModel);
            
            // Load templates
            if (File.Exists(templatesPath))
            {
                var templatesJson = File.ReadAllText(templatesPath);
                var templates = JsonConvert.DeserializeObject<dynamic>(templatesJson);
                
                _parentTemplates = JsonConvert.DeserializeObject<Dictionary<string, List<ParentRecord>>>(
                    templates.ParentTemplates.ToString());
                    
                _childTemplates = JsonConvert.DeserializeObject<Dictionary<string, List<ChildRecord>>>(
                    templates.ChildTemplates.ToString());
            }
            
            Console.WriteLine("Models and templates loaded successfully");
        }
        
        public ConsolidationPrediction Predict(RawDataInput input)
        {
            if (_parentPredictionEngine == null || _childPredictionEngine == null)
            {
                throw new InvalidOperationException("Models must be trained or loaded before making predictions");
            }
            
            // Create pattern key for this input
            string patternKey = $"{input.Category}_{input.Source}";
            
            // Make parent prediction
            var parentPrediction = _parentPredictionEngine.Predict(input);
            
            // Initialize the result
            var result = new ConsolidationPrediction
            {
                MessageId = input.MessageId,
                ConversionId = input.ConversionId,
                ShouldCreateParent = parentPrediction.ShouldCreateParent,
                ParentConfidence = parentPrediction.Probability,
                PredictedChildCount = 0,
                PredictedParent = null,
                PredictedChildren = new List<ChildRecord>()
            };
            
            // If we should create a parent, predict child count and content
            if (parentPrediction.ShouldCreateParent)
            {
                // Predict child count
                var childPrediction = _childPredictionEngine.Predict(input);
                result.PredictedChildCount = Math.Max(1, (int)Math.Round(childPrediction.PredictedChildCount));
                
                // Generate parent content based on templates
                result.PredictedParent = GenerateParentContent(input, patternKey);
                
                // Generate child content based on templates
                result.PredictedChildren = GenerateChildContent(input, patternKey, result.PredictedChildCount, result.PredictedParent?.ParentId);
            }
            
            return result;
        }
        
        private ParentRecord GenerateParentContent(RawDataInput input, string patternKey)
        {
            // Try to find a template matching the pattern
            if (_parentTemplates.ContainsKey(patternKey) && _parentTemplates[patternKey].Count > 0)
            {
                // Select a random template as the base
                Random random = new Random();
                var template = _parentTemplates[patternKey][random.Next(_parentTemplates[patternKey].Count)];
                
                // Clone the template
                var parent = new ParentRecord
                {
                    ParentId = $"P-{Guid.NewGuid():N}",
                    Name = template.Name,
                    Category = input.Category,
                    Status = template.Status,
                    CreatedDate = DateTime.Now,
                    TotalAmount = input.AmountValue // Use the input amount as a base
                };
                
                return parent;
            }
            
            // Fallback if no matching template is found
            return new ParentRecord
            {
                ParentId = $"P-{Guid.NewGuid():N}",
                Name = $"Auto-generated from {input.ConversionId}",
                Category = input.Category,
                Status = "New",
                CreatedDate = DateTime.Now,
                TotalAmount = input.AmountValue
            };
        }
        
        private List<ChildRecord> GenerateChildContent(RawDataInput input, string patternKey, int childCount, string parentId)
        {
            var children = new List<ChildRecord>();
            
            // Try to find templates matching the pattern
            if (_childTemplates.ContainsKey(patternKey) && _childTemplates[patternKey].Count > 0)
            {
                Random random = new Random();
                
                // Calculate how to divide the amount among children
                float totalAmount = input.AmountValue;
                float[] amounts = GenerateRandomDistribution(totalAmount, childCount);
                
                for (int i = 0; i < childCount; i++)
                {
                    // Select a random template
                    var template = _childTemplates[patternKey][random.Next(_childTemplates[patternKey].Count)];
                    
                    // Create a child record based on the template
                    var child = new ChildRecord
                    {
                        ChildId = $"C-{Guid.NewGuid():N}",
                        ParentId = parentId,
                        Amount = amounts[i],
                        Type = template.Type,
                        Description = input.Description ?? template.Description,
                        TransactionDate = DateTime.Now.AddDays(-random.Next(0, 5)) // Random date within last 5 days
                    };
                    
                    children.Add(child);
                }
            }
            else
            {
                // Fallback if no matching templates
                float amountPerChild = input.AmountValue / childCount;
                
                for (int i = 0; i < childCount; i++)
                {
                    var child = new ChildRecord
                    {
                        ChildId = $"C-{Guid.NewGuid():N}",
                        ParentId = parentId,
                        Amount = amountPerChild,
                        Type = "Standard",
                        Description = input.Description ?? $"Auto-generated child {i+1}",
                        TransactionDate = DateTime.Now
                    };
                    
                    children.Add(child);
                }
            }
            
            return children;
        }
        
        private float[] GenerateRandomDistribution(float total, int parts)
        {
            if (parts <= 0) return new float[0];
            if (parts == 1) return new float[] { total };
            
            Random random = new Random();
            float[] distribution = new float[parts];
            float sum = 0;
            
            // Generate random values
            for (int i = 0; i < parts; i++)
            {
                distribution[i] = (float)random.NextDouble();
                sum += distribution[i];
            }
            
            // Normalize to the total
            for (int i = 0; i < parts; i++)
            {
                distribution[i] = (distribution[i] / sum) * total;
            }
            
            return distribution;
        }
        
        public List<ConsolidationPrediction> PredictBatch(List<RawDataInput> inputs)
        {
            return inputs.Select(Predict).ToList();
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            // Example of using the consolidation engine
            try
            {
                var engine = new ConsolidationEngine();
                
                // Either train new models
                string trainingDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "training_data.csv");
                engine.TrainModels(trainingDataPath);
                
                string parentModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "parent_model.zip");
                string childModelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "child_model.zip");
                string templatesPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "content_templates.json");
                
                Directory.CreateDirectory(Path.GetDirectoryName(parentModelPath));
                engine.SaveModels(parentModelPath, childModelPath, templatesPath);
                
                // Or load existing models
                // engine.LoadModels(parentModelPath, childModelPath, templatesPath);
                
                // Make a prediction on a single record
                var sampleInput = new RawDataInput
                {
                    MessageId = "MSG12345",
                    ConversionId = "CONV789",
                    AmountValue = 1250.50f,
                    TransactionAge = 3.5f,
                    Category = "TypeA",
                    Source = "System1",
                    Description = "Sample transaction"
                };
                
                var prediction = engine.Predict(sampleInput);
                
                Console.WriteLine($"Message ID: {prediction.MessageId}");
                Console.WriteLine($"Conversion ID: {prediction.ConversionId}");
                Console.WriteLine($"Should create parent: {prediction.ShouldCreateParent} (Confidence: {prediction.ParentConfidence:P2})");
                
                if (prediction.ShouldCreateParent)
                {
                    Console.WriteLine($"Predicted child count: {prediction.PredictedChildCount}");
                    Console.WriteLine($"Parent ID: {prediction.PredictedParent.ParentId}");
                    Console.WriteLine($"Parent Amount: {prediction.PredictedParent.TotalAmount:C}");
                    Console.WriteLine($"Parent Category: {prediction.PredictedParent.Category}");
                    
                    Console.WriteLine("\nPredicted children:");
                    foreach (var child in prediction.PredictedChildren)
                    {
                        Console.WriteLine($"Child ID: {child.ChildId}, Amount: {child.Amount:C}, Type: {child.Type}");
                    }
                }
                
                // Process a batch of raw data
                Console.WriteLine("\nProcessing a batch of records...");
                var batchInputs = new List<RawDataInput>
                {
                    new RawDataInput { MessageId = "MSG1001", ConversionId = "CONV101", AmountValue = 500f, TransactionAge = 1.2f, Category = "TypeB", Source = "System2", Description = "First transaction" },
                    new RawDataInput { MessageId = "MSG1002", ConversionId = "CONV102", AmountValue = 1800f, TransactionAge = 5.7f, Category = "TypeA", Source = "System1", Description = "Second transaction" },
                    new RawDataInput { MessageId = "MSG1003", ConversionId = "CONV101", AmountValue = 350f, TransactionAge = 2.1f, Category = "TypeC", Source = "System3", Description = "Third transaction" }
                };
                
                var batchResults = engine.PredictBatch(batchInputs);
                
                foreach (var result in batchResults)
                {
                    Console.WriteLine($"Message: {result.MessageId}, Create Parent: {result.ShouldCreateParent}, Child Count: {result.PredictedChildCount}");
                    if (result.ShouldCreateParent)
                    {
                        Console.WriteLine($"  Parent: {result.PredictedParent.Name} ({result.PredictedParent.ParentId})");
                        Console.WriteLine($"  Children: {string.Join(", ", result.PredictedChildren.Select(c => c.ChildId))}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
            
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}