import streamlit as st
import pickle

# Load your trained models
with open('CCFD_model.pkl', 'rb') as file:
    fitted_models_and_predictions_dictionary = pickle.load(file)

def predict_fraud(input_data):
    predictions = {}
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        classifier = model_and_predictions['classifier']
        prediction = classifier.predict_proba([list(input_data.values())])[0][1]
        predictions[classifier_name] = prediction
    return predictions

# Streamlit UI
st.title("Fraud Detection Model")

# Inputs from user
tx_amount = st.number_input('Transaction Amount', value=300)
tx_during_weekend = st.selectbox('Transaction During Weekend?', [0, 1])
tx_during_night = st.selectbox('Transaction During Night?', [0, 1])
customer_nb_tx_1day_window = st.number_input('Customer TX 1-day window', value=5)
customer_avg_amount_1day_window = st.number_input('Customer Avg Amount 1-day window', value=5)
customer_nb_tx_7day_window = st.number_input('Customer TX 7-day window', value=20)
customer_avg_amount_7day_window = st.number_input('Customer Avg Amount 7-day window', value=60)
customer_nb_tx_30day_window = st.number_input('Customer TX 30-day window', value=80)
customer_avg_amount_30day_window = st.number_input('Customer Avg Amount 30-day window', value=70)
terminal_nb_tx_1day_window = st.number_input('Terminal TX 1-day window', value=10)
terminal_risk_1day_window = st.number_input('Terminal Risk 1-day window', value=0.2)
terminal_nb_tx_7day_window = st.number_input('Terminal TX 7-day window', value=40)
terminal_risk_7day_window = st.number_input('Terminal Risk 7-day window', value=0.3)
terminal_nb_tx_30day_window = st.number_input('Terminal TX 30-day window', value=150)
terminal_risk_30day_window = st.number_input('Terminal Risk 30-day window', value=0.4)

# Predict button
if st.button("Predict"):
    input_data = {
        'TX_AMOUNT': tx_amount,
        'TX_DURING_WEEKEND': tx_during_weekend,
        'TX_DURING_NIGHT': tx_during_night,
        'CUSTOMER_ID_NB_TX_1DAY_WINDOW': customer_nb_tx_1day_window,
        'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW': customer_avg_amount_1day_window,
        'CUSTOMER_ID_NB_TX_7DAY_WINDOW': customer_nb_tx_7day_window,
        'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW': customer_avg_amount_7day_window,
        'CUSTOMER_ID_NB_TX_30DAY_WINDOW': customer_nb_tx_30day_window,
        'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW': customer_avg_amount_30day_window,
        'TERMINAL_ID_NB_TX_1DAY_WINDOW': terminal_nb_tx_1day_window,
        'TERMINAL_ID_RISK_1DAY_WINDOW': terminal_risk_1day_window,#its in point 0.2
        'TERMINAL_ID_NB_TX_7DAY_WINDOW': terminal_nb_tx_7day_window,
        'TERMINAL_ID_RISK_7DAY_WINDOW': terminal_risk_7day_window, #in point 0.3
        'TERMINAL_ID_NB_TX_30DAY_WINDOW': terminal_nb_tx_30day_window,
        'TERMINAL_ID_RISK_30DAY_WINDOW': terminal_risk_30day_window,
    }
    predictions = predict_fraud(input_data)
    st.write("Fraud Predictions:", predictions)