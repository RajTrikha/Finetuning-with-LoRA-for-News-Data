# Deep Learning Project 2: Text Classification with RoBERTa and LoRA

## Team Members
Solomon Martin, Raj Trikha, Gokuleshwaran

## Project Description
This project implements fine-tuning of a pre-trained RoBERTa model with Low-Rank Adaptation (LoRA) for text classification on the AG News dataset. The primary objective was to develop an efficient classifier that maintains high accuracy while reducing the number of trainable parameters by using parameter-efficient fine-tuning techniques.

## Project Achievements
- Successfully implemented LoRA adapters on RoBERTa for text classification
- Experimented with various LoRA configurations (rank and alpha values)
- Applied effective training strategies including OneCycleLR scheduler and mixed-precision training
- Created an ensemble of multiple model configurations for improved accuracy

## Data
The project uses two datasets:
- AG News: A collection of news articles categorized into 4 classes (World, Sports, Business, Sci/Tech)
- Custom test dataset: An unlabeled test set provided for evaluation in the competition

## Project Structure
```
project/
├── project2-95d784-2.ipynb  # Main notebook with training code
└── README.md                # Project documentation
```

## Approach
1. **Model Selection**: Used RoBERTa-base as the foundation model for its strong performance on NLP tasks
2. **Parameter-Efficient Fine-Tuning**: Applied LoRA to reduce trainable parameters while maintaining performance
3. **Training Optimization**:
   - Implemented cosine learning rate schedule
   - Applied weight decay and warmup to improve convergence
   - Configured effective batch sizes and training epochs
4. **Ensemble Method**: Combined predictions from multiple models to boost final accuracy

## Technical Implementation
- **Base Model**: RoBERTa from Hugging Face Transformers
- **Fine-tuning Method**: LoRA with different rank configurations (r=4, 8, 16)
- **Training Framework**: Hugging Face Trainer with custom optimizers
- **Hardware**: GPU acceleration for faster training

## Running the Project
1. Install dependencies:
```bash
pip install transformers datasets peft accelerate torch numpy pandas
```

2. Load and preprocess the datasets:
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
```

3. Configure and train the model:
```python
from transformers import AutoModelForSequenceClassification, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# Load model and apply LoRA
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["query", "value"])
model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(model=model, ...)
trainer.train()
```

4. Generate predictions on test data:
```python
# Run inference and save predictions
predictions = trainer.predict(test_dataset)
results = pd.DataFrame({"ID": range(len(predictions)), "Label": np.argmax(predictions.predictions, axis=1)})
results.to_csv("submission.csv", index=False)
```

## Key Findings
- LoRA significantly reduced training parameters while maintaining high accuracy
- Different LoRA configurations (r and alpha values) showed varying performance
- Ensemble methods proved effective for improving final predictions
- Custom learning rate schedules improved model convergence

## Acknowledgements
This project was developed as part of the Deep Learning course at New York University. We thank the course instructors for their guidance and support.
