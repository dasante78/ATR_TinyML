
# Tiny ML Train

This project explores methods to optimize sequence-based language models by selectively "forgetting" parts of sequences. The goal is to reduce the sequence length and computation cost while preserving key information, achieving computational savings of \(O(n^2 - b^2)\), where \(b < n\).

## Features

- **Forgetting Strategies**: 
  - Random Forget Mask
  - MLP-Based Forget Mask
  - CNN-MLP Forget Mask
  - RNN Forget Mask
  - Identity Mask (baseline)

- **Reinforcement Learning**: 
  - Train a base language model alongside forgetting layers using reinforcement-based exploration and exploitation strategies.

- **Token Pruning and Masking**: 
  - Apply forgetting masks to input sequences.
  - Dynamically reduce sequence length during training.

- **Integration with HuggingFace Transformers**:
  - Use pre-trained language models as the base for forgetting experiments.

## Dependencies

Install the required Python packages:

```bash
pip install huggingface_hub spacy datasets wandb
python -m spacy download en_core_web_sm
```

## Setup

1. Clone the repository and set up your environment.
2. Log in to HuggingFace and WandB:
   ```python
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")
   import wandb
   wandb.login(key="YOUR_WANDB_KEY")
   ```
3. Download and preprocess the dataset:
   - Default: `wikitext-103-raw-v1`
   - Preprocessed using the tokenizer corresponding to the chosen base model.

## Usage

### Training

Select a forgetting strategy and configure the training parameters:

1. Choose a forget mask:
   ```python
   mask = IdentityMask()
   mask = RandomForgetMask(forget_prob=0.1)
   mask = MLPForgetMask(base_llm_model)
   mask = RNNForgetMask(base_llm_model, bidirectional=False)
   ```

2. Train the model with reinforcement learning:
   ```python
   reinforce_model(base_llm_model, mask, forget_p=0.1, exploration_epochs=[0.75, 0.5, 0.25, 0], ...)
   ```

3. Alternative: Two-step reinforcement learning:
   ```python
   two_step_reinforce_model(base_llm_model, mask, forget_p=0.1, exploration_epochs=[True, False, ...], ...)
   ```

### Helper Functions

- `create_random_forgetting_mask(input_ids, forget_prob)`: Creates random masks for token forgetting.
- `apply_mask(input_ids, mask)`: Applies masks and pads sequences to uniform length.

## Results and Logging

- Track training progress and metrics using [WandB](https://wandb.ai/).
- Models and forgetting layers are saved periodically.

## Evaluation

The evaluation process is designed to assess the performance of the optimized language models on specific tasks after applying the forgetting strategies.

### Evaluation Features
- Measure model accuracy and efficiency post-forgetting.
- Compare performance metrics (e.g., loss, perplexity) with baseline models.
- Assess computational savings in terms of runtime and memory.

### Implementation
The `eval` notbook includes functionalities for:
- Loading the trained model and dataset.
- Computing evaluation metrics on test datasets.
- Comparing results with baseline models to quantify improvements.

## Future Work

- Extend the forgetting strategies to incorporate more advanced architectures.
- Experiment with various datasets and pre-trained language models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project utilizes HuggingFace Transformers and WandB for efficient experimentation and logging.
