# LLM-fraud-intent-crypto
This repository contains a simple implementation of the annotation pipeline
described in our paper:

"Detecting Fraud Intent in Cryptocurrency Discussions with LLM-Assisted Annotation"

The script uses a local large language model via Ollama
to classify text comments into three categories:

- Fraud intention
- Solution or prevention intention
- Out of context

Datasets are **not** included for privacy and licensing reasons.
Users must provide their own CSV or Excel file with a text column.

---

## LLM-assisted labeling script

The file `llm_labeling/run_labeling.py` implements our annotation pipeline.
To use it, install the dependencies (`pandas`, `tqdm`, `ollama`) and configure
a local model in Ollama (for example `mistral:instruct`).

Then edit the parameters in the `__main__` block of `run_labeling.py`:

- `INPUT_FILE`: path to your CSV/Excel file  
- `INPUT_SHEET`: sheet name for Excel files (or `None`)  
- `TEXT_COLUMN`: name of the text column to classify  
- `OUTPUT_FILE`: path for the output CSV with labels  
- `MODEL_NAME`: Ollama model to use  
- `PROMPT_CHOICE`: `baseline`, `domain_specific`, or `intent_focused`  

The script produces a labeled dataset that can be used for downstream machine learning tasks.

---

## Training a machine learning classifier

Once the dataset has been labeled, the file `ml_classifier.py` can be used to train
a text classification model on the annotated data.

The script implements a simple text classification pipeline:

1. Load the labeled dataset (`text`, `label`)
2. Split the data into training and test sets
3. Convert text into TF-IDF features
4. Train several baseline classifiers
5. Evaluate the models using standard metrics
6. Save the best-performing model and the TF-IDF vectorizer

The pipeline currently trains three baseline machine learning classifiers:

- Multinomial Naive Bayes  
- Logistic Regression  
- Random Forest  

After evaluation, the script automatically selects the **best-performing model based on the F1-score** and saves it together with the TF-IDF vectorizer.

### Model selection and extensions

This implementation is intended as a **baseline example**. Users are encouraged to further improve performance by:

- tuning model hyperparameters (e.g., grid search or cross-validation)
- experimenting with additional machine learning models
- modifying the TF-IDF configuration or feature extraction strategy

Depending on the specific classification task and dataset characteristics, other classifiers (such as SVM, gradient boosting methods, or neural models) may yield better results.

NOTE: Experimental results are reported in the paper.
