# stoten_1_sloneczne

# Urban Lake Water Quality Prediction

This project uses various Machine Learning techniques to predict the water quality of an urban lake, based on 20 years of collected data. The algorithms used include Multiple Linear Regression, Artificial Neural Networks, and Random Forest.

## Data

The water quality data has been collected over a period of two decades, and includes parameters such as Dissolved Oxygen (DO), Biological Oxygen Demand (BOD), pH, Turbidity, Nitrate, and Phosphate. The catchment area of the lake covers 129 km², including residential areas, fields and recreational plots, green areas, and production areas. Sewage discharge locations have been identified, especially those originating from fields, recreational plots, and production plants.

## Code Structure

- `libraries.py`: Contains all the necessary library imports for the project.
- `fut_importance.py`: Code for calculating and displaying the feature importance for each machine learning model.
- `stat.py`: Contains functions for descriptive statistics, as well as normality and stationarity tests.
- `vis.py`: Functions for generating various visualizations such as correlation matrices, time series plots, and heatmaps.
- `main.py`: The main script that runs the entire pipeline, from data loading and preprocessing to model training, evaluation, and interpretation.

## Getting Started

To run this project, you first need to clone this repository to your local machine. Then, navigate to the project directory and install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can run the main script by:
```python main.py```


## License

This project is licensed under the MIT License.

Please adapt this template according to your specific project details, and remember to replace the placeholders with actual information. For example, you might want to add more details about the data, the machine learning models used, the results obtained, and how to interpret these results.

