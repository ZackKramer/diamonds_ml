# App to predict the prices of diamonds using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt 


# Set up the app title and image
st.title('Diamond Price Predictor')
st.write("This app helps you estimate prices based on selected features")
st.image('diamond_image.jpg', use_column_width = True)

alpha = st.slider('Select alpha value for prediction intervals',
          min_value = 0.01,
          max_value = 0.5,
          value = 0.1,
          step = 0.01)

# Reading the pickle file that we created before 
model_pickle = open('reg_diamonds.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('diamonds.csv')

# Sidebar for user inputs with an expander
with st.sidebar.form("user_feature_input",):
    st.image('diamond_sidebar.jpg', use_column_width = True,
             caption = "Diamond Price Predictor")
    st.header("Diamond Features Input Input")
    st.write("You can either upload your data file or manually enter diamond features")
    with st.expander("Option 1: Upload CSV File"):
        st.header("Upload a CSV file containing the diamond details.")
        diamond_file = st.file_uploader("Choose a CSV file")
        st.header("Sample Data Format for Upload")
        st.dataframe(default_df.head())
        st.write("Ensure your uploaded file has the same column names and data types as shown above.")
    with st.expander("Option 2: Fill Out Form"):
        st.header("Enter the diamond details manually using the form below.")
        cut = st.selectbox('Cut Quality', 
                                         options = default_df['cut'].unique())
        color = st.selectbox('Diamond Color',
                                        options = default_df['color'].unique())
        clarity = st.selectbox('Clarity',
                                     options = default_df['clarity'].unique())
        carat = st.number_input('Carat Weight',
                                min_value = default_df['carat'].min(),
                                max_value = default_df['carat'].max(),
                                value = 1.0,
                                step = .01)
        depth = st.number_input('Depth (%)',
                                min_value = default_df['depth'].min(),
                                max_value = default_df['depth'].max(),
                                value = 50.0,
                                step = .1)
        table = st.number_input('Table (%)',
                                min_value = default_df['table'].min(),
                                max_value = default_df['table'].max(),
                                value = 50.0,
                                step = 1.0)
        x = st.number_input('Length (mm)',
                                min_value = default_df['x'].min(),
                                max_value = default_df['x'].max(),
                                value = 5.0,
                                step = .01)
        y = st.number_input('Width (mm)',
                                min_value = default_df['y'].min(),
                                max_value = default_df['y'].max(),
                                value = 5.0,
                                step = .01)
        z = st.number_input('Depth (mm)',
                                min_value = default_df['z'].min(),
                                max_value = default_df['z'].max(),
                                value = 5.0,
                                step = .01)
        submit_button = st.form_submit_button("Submit Form Data")

if diamond_file is None:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['price'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [carat,
                                    cut,
                                    color,
                                    clarity,
                                    depth,
                                    table,
                                    x,
                                    y,
                                    z]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]

    # Ensure limits are within [0, 10000]
    lower_limit = max(0, lower_limit[0][0])
    upper_limit = min(10000, upper_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Prices...")

    # Display results using metric card
    st.metric(label = "Predicted Price", value = f"${pred_value:.2f}")
    st.write(f"**Confidence Interval** ({100-alpha*100}%): [{lower_limit:.2f}, {upper_limit:.2f}]")

else:
    # Loading data
    user_df = pd.read_csv(diamond_file) # User provided data
    original_df = pd.read_csv('diamonds.csv') # Original data to create ML model
    
    # Dropping null values
    user_df = user_df.dropna() 
    original_df = original_df.dropna() 
    
    # Remove output (species) and year columns from original data
    original_df = original_df.drop(columns = ['price'])
    
    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred, user_intervals = reg_model.predict(user_df_encoded, alpha = alpha)

    # Predicted prices
    user_pred_species = user_pred
    user_lower_limit = user_intervals[:, 0].round(2)
    user_upper_limit = user_intervals[:, 1].round(2)

    # Adding predicted species to user dataframe
    user_df['Predicted Price'] = user_pred_species
    user_df['Lower Price Limit'] = user_lower_limit
    user_df['Upper Price Limit'] = user_upper_limit
    
    user_df['Lower Price Limit'] = user_df['Lower Price Limit'].apply(lambda x: max(0, x))
    user_df['Upper Price Limit'] = user_df['Upper Price Limit'].apply(lambda x: min(10000, x))

    # Show the predicted species on the app
    st.subheader(f"Prediction Results with Confidence Interval of {100-alpha*100}%")
    st.dataframe(user_df)

# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
