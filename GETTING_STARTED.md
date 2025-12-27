# 📝 Instructions for Getting Started

## Step 1: Download the Dataset

1. Go to Kaggle and search for "Student Mental Health" dataset
2. Download the CSV file
3. Place it in `data/raw/student_mental_health.csv`

**Alternative datasets** (if the above is not available):
- Search Kaggle for student dropout or academic performance datasets
- Ensure the dataset has dropout indicators and student features

## Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Run Jupyter Notebooks (IN ORDER)

```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Execution Order:
1. `notebooks/01_data_exploration.ipynb` - Explore the dataset
2. `notebooks/02_data_preprocessing.ipynb` - Clean and prepare data (TO BE CREATED)
3. `notebooks/03_baseline_models.ipynb` - Train initial models (TO BE CREATED)
4. `notebooks/04_model_optimization.ipynb` - Tune hyperparameters (TO BE CREATED)
5. `notebooks/05_model_interpretability.ipynb` - SHAP/LIME explanations (TO BE CREATED)

## Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Step 5: Make Predictions

After training the models, you can use the "Predict" page to assess dropout risk for individual students.

## Troubleshooting

### Dataset Not Found
- Ensure the CSV file is in `data/raw/student_mental_health.csv`
- Check the file name matches exactly

### Module Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

### Notebook Kernel Issues
- Select the correct kernel (venv environment) in Jupyter
- Restart the kernel if needed

## Next Steps

After running notebook 01:
1. Review the data exploration results
2. Identify the target variable (dropout indicator)
3. Note important features for preprocessing
4. Update configuration in `config.py` if needed

---

**Need Help?** Check the README.md for more details or review the implementation plan.
