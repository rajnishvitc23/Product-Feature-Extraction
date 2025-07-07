# Product Feature Extraction & Sentiment Grouping (FLAN-T5 Few-Shot)

## Overview

This Streamlit app allows you to upload a CSV file of product reviews, automatically extract product features using the FLAN-T5 language model, group features by sentiment (Positive, Neutral, Negative) based on review ratings, and visualize the results with interactive charts and summary statistics.

Deployed here:: 

[[https://app-starter-kit.streamlit.app/](https://feature-extraction-flan-t5.streamlit.app/
](https://feature-extraction-flan-t5.streamlit.app/)

## Features

- **Upload CSV** of product reviews.
- **Extract product features** from each review using few-shot prompting with FLAN-T5.
- **Assign sentiment** to each review based on its rating (Positive, Neutral, Negative).
- **Aggregate and group features** by sentiment.
- **Visualize**:
  - Most mentioned product features (bar chart).
  - Review ratings distribution (bar chart).
  - Summary statistics (unique features, most/least frequent, average rating).
- **Download** the results as a CSV file.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### 2. Install Requirements

Create a `requirements.txt` file with the following content:

```
streamlit
pandas
transformers
torch
spacy
matplotlib
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Prepare a CSV file with at least these columns:

| Review ID | Product Name | Review Text                        | Rating |
|-----------|-------------|------------------------------------|--------|
| 1         | Phone X     | The battery life is excellent.     | 5      |
| 2         | Phone X     | Camera quality is disappointing.   | 2      |
| ...       | ...         | ...                                | ...    |

- **Review Text**: The column containing the review content.
- **Rating**: Numerical rating (1–5).

### 4. Run the App

```bash
streamlit run app.py
```

## Usage

1. **Upload your CSV file** using the file uploader.
2. **Select the review and rating columns** as prompted.
3. Click **"Extract Features and Group by Sentiment"**.
4. View:
   - Features grouped by sentiment.
   - Most mentioned features (bar chart).
   - Review ratings distribution (bar chart).
   - Summary statistics.
5. **Download** the processed results as a CSV.

## How It Works

- Uses **FLAN-T5** (via Hugging Face Transformers) with few-shot prompting to extract aspect phrases from each review.
- Uses **spaCy** to reduce aspect phrases to their core feature (main noun).
- Assigns sentiment based on the review's rating:
  - Ratings 4–5: Positive
  - Rating 3: Neutral
  - Ratings 1–2: Negative
- Aggregates and visualizes features by sentiment and frequency.

## Notes

- The first run may take a few minutes to download the FLAN-T5 model and spaCy language model.
- For large datasets, processing may take longer due to model inference time.
- For deployment on Streamlit Community Cloud, all dependencies and model downloads are handled automatically.

## Troubleshooting

- **Memory errors:** Try using a smaller FLAN-T5 model (e.g., `flan-t5-small`) if you encounter resource issues.
- **Missing spaCy model:** The app will attempt to download the English model automatically if not found.
- **Slow performance:** For large files, consider testing locally before deploying to the cloud.

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgements

- [Google FLAN-T5](https://huggingface.co/google/flan-t5-base)
- [spaCy NLP Toolkit](https://spacy.io/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Author

Rajnish Mandavariya
[Your GitHub Profile](https://github.com/rajnishvitc23)
