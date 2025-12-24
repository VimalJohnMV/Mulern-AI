# README.md

## Project Title: Annual Temperature Prediction using Recurrent Neural Networks

## Overview
This project demonstrates the application of a Recurrent Neural Network (RNN), specifically an LSTM model, to predict future annual temperature conditions based on historical seasonal temperature data for India. The process involves data loading, comprehensive preprocessing, model building, training with early stopping, evaluation, and visualization of predictions.

## Dataset
*   **Source**: The dataset was downloaded from KaggleHub: `naivedatamodel/temperature-dataset-india`.
*   **Description**: The dataset `TEMP_ANNUAL_SEASONAL_MEAN.csv` contains 123 entries with annual and seasonal mean temperature data for India. Key columns include `YEAR`, `ANNUAL`, `JAN-FEB`, `MAR-MAY`, `JUN-SEP`, and `OCT-DEC` temperatures.

## Data Preprocessing
1.  **Data Type Conversion**: Seasonal temperature columns were converted from `object` to `float64`, coercing non-numeric values to `NaN`.
2.  **Missing Value Handling**: Missing values were handled using linear interpolation, followed by backward-fill (`bfill`) and forward-fill (`ffill`) to ensure no `NaN` values remained.
3.  **Feature Scaling**: Relevant temperature features (`ANNUAL`, `JAN-FEB`, `MAR-MAY`, `JUN-SEP`, `OCT-DEC`) were scaled to a range of (0, 1) using `MinMaxScaler`.
4.  **Sequence Creation**: Time-series sequences were created with a `look_back` window of `3`, meaning 3 previous timesteps were used to predict the next `ANNUAL` temperature.
5.  **Train-Test Split**: The data was split into 80% for training and 20% for testing, maintaining temporal order.

## Model Architecture
*   **Type**: Recurrent Neural Network (RNN) using an LSTM layer.
*   **Framework**: Developed with TensorFlow and Keras.
*   **Layers**:
    *   `LSTM` layer with 50 units, receiving input sequences of shape `(3, 5)` (timesteps, features).
    *   `Dense` output layer with 1 unit to predict the `ANNUAL` temperature.
*   **Total Trainable Parameters**: 11,251.

## Training
*   **Optimizer**: Adam
*   **Loss Function**: Mean Squared Error (`mean_squared_error`)
*   **Metrics**: Mean Absolute Error (`mae`)
*   **Epochs**: Initial training ran for 100 epochs, but the model was later retrained with Early Stopping.
*   **Early Stopping**: An `EarlyStopping` callback was implemented to monitor `val_loss` with a `patience` of 10 epochs. `restore_best_weights` was set to `True`. This prevented overfitting and optimized training duration, with the model stopping at Epoch 12 during retraining.

## Results and Evaluation
After retraining with early stopping:
*   **Test Loss (MSE)**: `0.0474`
*   **Test MAE**: `0.1985`

**Interpretation**: The model exhibits reasonable predictive accuracy for annual temperature, with predictions deviating by approximately 0.1985 degrees Celsius from actual values on average. The application of early stopping improved generalization, resulting in lower test loss and MAE compared to the initial training without early stopping.

## Visualizations
1.  **Actual vs. Predicted Annual Temperature**: A plot comparing the model's predicted annual temperatures against the actual annual temperatures from the test set. This visualization shows that the model generally captures the trend of the actual data, indicating its ability to learn underlying patterns.
2.  **Model Loss over Epochs**: A plot displaying the training loss and validation loss across epochs. This plot was used to identify potential overfitting, showing that validation loss stabilized while training loss continued to decrease, leading to the implementation of early stopping.

## Dependencies
*   `kagglehub`
*   `pandas`
*   `numpy`
*   `sklearn` (for `MinMaxScaler`)
*   `tensorflow` (for `keras.models.Sequential`, `keras.layers.LSTM`, `keras.layers.Dense`, `keras.callbacks.EarlyStopping`)
*   `matplotlib`
