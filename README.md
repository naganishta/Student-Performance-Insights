# Student Insights — Student Performance Analytics

## Included features
- Interactive dashboard KPIs and filters for all five categorical attributes
- Average-score distribution, subject comparison, box plot and performance-level breakdown
- Individual student explorer, top 10 performers and single-record CSV export
- Group comparisons across gender, ethnicity, parental education, lunch and test preparation
- Correlation heatmap and interactive subject scatter plots
- What-if score calculator
- Random Forest average-score estimator with held-out evaluation metrics
- Data-quality checks, descriptive statistics, filtered/full cleaned CSV exports
- PDF summary report

## Run locally
1. Extract the ZIP file.
2. Open a terminal in the extracted folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start Streamlit:
   `streamlit run app.py`

The app automatically loads `StudentsPerformance.csv` from the same folder. You can also upload another compatible CSV from the sidebar.

## Dataset columns required
gender, race/ethnicity, parental level of education, lunch, test preparation course,
math score, reading score, writing score.

## Important notes
- The source dataset has no student names or permanent IDs. The app assigns row numbers for navigation only.
- Performance categories use transparent descriptive cutoffs: below 40 = Needs Support, 40–69.99 = Developing, 70–100 = High.
- The ML estimator is a learning demonstration. It estimates average score from demographic/course attributes and should not be used for high-stakes decisions.
- Correlation and group differences do not prove causation.
