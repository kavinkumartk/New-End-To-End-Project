import gradio as gr
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

model = None

def load_model():
    """Load model from MLflow or create fallback"""
    global model
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        experiment = mlflow.get_experiment_by_name("satisfaction_classification")
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        gradient_boosting_runs = runs[runs['tags.mlflow.runName'] == 'GradientBoosting']
        if not gradient_boosting_runs.empty:
            best_run = gradient_boosting_runs.loc[gradient_boosting_runs['metrics.f1'].idxmax()]
            run_id = best_run['run_id']
            model_uri = f"runs:/{run_id}/GradientBoosting"
            model = mlflow.sklearn.load_model(model_uri)
            print("Model loaded from MLflow")
            return True
    except Exception as e:
        print(f"MLflow error: {e}")
    
    try:
        df = pd.read_csv('trainscaled.csv')
        y = df['satisfaction']
        X = df.drop(columns=['satisfaction'])
        
        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Fallback model created with proper preprocessing")
        return True
    except Exception as e:
        print(f"Fallback error: {e}")
        return False

def predict_satisfaction(gender, customer_type, age, travel_type, class_type, flight_distance,
                        wifi, departure_convenient, online_booking, gate_location, food_drink,
                        online_boarding, seat_comfort, entertainment, on_board_service,
                        leg_room, baggage_handling, checkin_service, inflight_service,
                        cleanliness, departure_delay, arrival_delay):
    """Predict customer satisfaction"""
    if model is None:
        return "Model not loaded"
    
    try:
        data = {
            'Gender': gender,
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': travel_type,
            'Class': class_type,
            'Flight Distance': flight_distance,
            'Inflight wifi service': wifi,
            'Departure/Arrival time convenient': departure_convenient,
            'Ease of Online booking': online_booking,
            'Gate location': gate_location,
            'Food and drink': food_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': entertainment,
            'On-board service': on_board_service,
            'Leg room service': leg_room,
            'Baggage handling': baggage_handling,
            'Checkin service': checkin_service,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            'Departure Delay in Minutes': departure_delay,
            'Arrival Delay in Minutes': arrival_delay
        }
        
        df = pd.DataFrame([data])
        
        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        prediction = model.predict(df.values)[0]
        prediction_label = "Satisfied" if prediction == 1 else "Not Satisfied"
        
        return prediction_label
    except Exception as e:
        return f"Error: {str(e)}"

print("Loading model...")
load_model()

print("Creating Gradio interface...")

inputs = [
    gr.Dropdown(["Male", "Female"], label="Gender", value="Female"),
    gr.Dropdown(["Loyal Customer", "disloyal Customer"], label="Customer Type", value="Loyal Customer"),
    gr.Number(label="Age", value=52),
    gr.Dropdown(["Business travel", "Personal Travel"], label="Type of Travel", value="Business travel"),
    gr.Dropdown(["Eco", "Business", "Eco Plus"], label="Class", value="Eco"),
    gr.Number(label="Flight Distance", value=160),
    gr.Slider(1, 5, step=1, value=5, label="Inflight WiFi Service"),
    gr.Slider(1, 5, step=1, value=4, label="Departure/Arrival Time Convenient"),
    gr.Slider(1, 5, step=1, value=3, label="Ease of Online Booking"),
    gr.Slider(1, 5, step=1, value=4, label="Gate Location"),
    gr.Slider(1, 5, step=1, value=3, label="Food and Drink"),
    gr.Slider(1, 5, step=1, value=4, label="Online Boarding"),
    gr.Slider(1, 5, step=1, value=3, label="Seat Comfort"),
    gr.Slider(1, 5, step=1, value=5, label="Inflight Entertainment"),
    gr.Slider(1, 5, step=1, value=5, label="On-board Service"),
    gr.Slider(1, 5, step=1, value=5, label="Leg Room Service"),
    gr.Slider(1, 5, step=1, value=5, label="Baggage Handling"),
    gr.Slider(1, 5, step=1, value=5, label="Check-in Service"),
    gr.Slider(1, 5, step=1, value=2, label="Inflight Service"),
    gr.Slider(1, 5, step=1, value=5, label="Cleanliness"),
    gr.Number(label="Departure Delay (Minutes)", value=50),
    gr.Number(label="Arrival Delay (Minutes)", value=44)
]
outputs = gr.Textbox(label="Prediction", interactive=False)

demo = gr.Interface(
    fn=predict_satisfaction,
    inputs=inputs,
    outputs=outputs,
    title="🛫 Customer Satisfaction Prediction",
    description="Predict whether a customer will be satisfied with their flight experience.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()