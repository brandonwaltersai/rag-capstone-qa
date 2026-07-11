# Data

This project does not vendor any third-party data — both datasets are pulled at
setup time via `kagglehub` and cached locally (git-ignored).

| Role | Dataset | Source |
|---|---|---|
| Knowledge base | Bitext Gen AI Chatbot Customer Support Dataset | [Kaggle](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset), published by Bitext |
| Evaluation utterances | Training Dataset for Chatbots / Virtual Assistants | [Kaggle](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants), published by Bitext |

Fetch both with:

```python
from src.data_prep import fetch_raw_csvs
kb_csv, utt_csv = fetch_raw_csvs()
```

Requires a Kaggle API token (`~/.kaggle/kaggle.json`) — see the
[Kaggle API docs](https://www.kaggle.com/docs/api).

Attribution: both datasets are published by Bitext on Kaggle for chatbot /
virtual-assistant training and evaluation. Refer to the dataset pages above
for their current license terms before redistributing derived data.
