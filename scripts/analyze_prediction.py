import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from sklearn.compose import ColumnTransformer

# Function to merge train/test with store data
def merge_data(train_file, test_file, store_file):
    """
    Merge the train/test and store data based on the 'Store' column.
    Accepts either file paths or DataFrames as inputs.
    """
    # Check if input is a file path, and read as DataFrame if necessary
    if isinstance(train_file, str):
        train_df = pd.read_csv(train_file, dtype={'StateHoliday': 'str'})
    else:
        train_df = train_file

    if isinstance(store_file, str):
        store_df = pd.read_csv(store_file)
    else:
        store_df = store_file

    # Merge train and store data
    train_merged = pd.merge(train_df, store_df, how='inner', on='Store')

    test_merged = None  # Default test_merged to None
    if test_file is not None:
        if isinstance(test_file, str):
            test_df = pd.read_csv(test_file)
        else:
            test_df = test_file

        # Merge test and store data
        test_merged = pd.merge(test_df, store_df, how='inner', on='Store')

    return train_merged, test_merged

# Feature Engineering: Extract new features from datetime columns
def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def convert_state_holiday(df):
    df['StateHoliday'] = df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
    return df

def encode_categorical_columns(train_df, test_df):
    categorical_cols = ['StoreType', 'Assortment', 'PromoInterval']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )

    # For the training dataset, drop 'Sales' and 'Customers'
    train_features = train_df.drop(columns=['Sales', 'Customers'], errors='ignore')
    
    # For the test dataset, drop 'Customers' if it exists
    test_features = test_df.drop(columns=['Customers'], errors='ignore')

    # Fit the encoder on the training data and transform both train and test
    train_df_encoded = preprocessor.fit_transform(train_features)
    test_df_encoded = preprocessor.transform(test_features)

    # Get feature names after encoding
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    encoded_cat_names = [name.replace(',', '_') for name in encoded_cat_names]
    numeric_cols = train_features.drop(columns=categorical_cols).columns
    final_feature_names = np.concatenate([encoded_cat_names, numeric_cols])

    return train_df_encoded, test_df_encoded, final_feature_names

def preprocess_data(train_df, test_df):
    train_df = create_date_features(train_df)
    test_df = create_date_features(test_df)

    # Drop the original 'Date' column after feature extraction
    train_df.drop(columns=['Date'], inplace=True)
    test_df.drop(columns=['Date'], inplace=True)

    train_df = convert_state_holiday(train_df)
    test_df = convert_state_holiday(test_df)

    # Handle Missing Values in Categorical Columns (e.g., 'StoreType', 'Assortment')
    train_df['StoreType'].fillna('unknown', inplace=True)
    test_df['StoreType'].fillna('unknown', inplace=True)

    train_df['Assortment'].fillna('unknown', inplace=True)
    test_df['Assortment'].fillna('unknown', inplace=True)

    train_df_encoded, test_df_encoded, final_feature_names = encode_categorical_columns(train_df, test_df)

    # Convert encoded data back into DataFrame format with proper column names
    train_df_encoded = pd.DataFrame(train_df_encoded, columns=final_feature_names)
    test_df_encoded = pd.DataFrame(test_df_encoded, columns=final_feature_names)

    # Add back the 'Sales' column to the processed train dataset
    if 'Sales' in train_df.columns:
        train_df_encoded['Sales'] = train_df['Sales'].values

    return train_df_encoded, test_df_encoded

# Model building function
def build_model(train_df, target_column):
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    if target_column not in train_df.columns:
        raise KeyError(f"'{target_column}' not found in DataFrame.")
    
    # Define features (X) and target (y)
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Convert all column names in X to strings
    X.columns = X.columns.astype(str)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f'Cross-validation RMSE: {(-cv_scores.mean()) ** 0.5}')

    # Predict on validation data
    y_pred = model_pipeline.predict(X_val)
    val_rmse = rmse(y_val, y_pred)
    print(f'Validation RMSE: {val_rmse}')

    return model_pipeline, X_train, X_val, y_train, y_val

# Prediction function
def prediction():
    """
    Main function to load data, preprocess, train, and save the model.
    """
    # Load data files into DataFrames
    train_file = pd.read_csv(r'c:\Users\Administrator\Desktop\KAIM\Rossmann_Pharmaceuticals_sales_forecast\data\train.csv')
    test_file = pd.read_csv(r'c:\Users\Administrator\Desktop\KAIM\Rossmann_Pharmaceuticals_sales_forecast\data\test.csv')
    store_file = pd.read_csv(r'c:\Users\Administrator\Desktop\KAIM\Rossmann_Pharmaceuticals_sales_forecast\data\store.csv')

    # Merge data
    train_merged, test_merged = merge_data(train_file, test_file, store_file)

    # Ensure 'Sales' column exists in the merged training data
    if 'Sales' not in train_merged.columns:
        raise KeyError("'Sales' column is missing in the merged training data.")

    # Preprocess data
    processed_train, processed_test = preprocess_data(train_merged, test_merged)

    # Ensure 'Sales' column exists in the processed training data
    if 'Sales' not in processed_train.columns:
        raise KeyError("'Sales' column is missing after preprocessing.")

    # Build and train the model
    model_pipeline, X_train, X_val, y_train, y_val = build_model(processed_train, target_column='Sales')

    # Save the trained model with a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"random_forest_model_{timestamp}.pkl"
    joblib.dump(model_pipeline, model_filename)

    return model_pipeline, X_train, X_val, y_train, y_val

