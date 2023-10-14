import gradio as gr
import joblib
import numpy as np
import pickle


model_path = "/xgboost-model.pkl"
with open(model_path, "rb") as f:
    my_model = pickle.load(f)
    
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
        
	feat_list = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]], dtype=object)
	return my_model.predict(feat_list)

def predict_death_event1(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets,
             serum_creatinine, serum_sodium, sex, smoking, time):
  x = np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets,
             serum_creatinine, serum_sodium, sex, smoking, time])
  prediction = my_model.predict(x.reshape(1, -1))

  return prediction

title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn = predict_death_event,
                         inputs=[
                            gr.Slider(2, 100, value=30, label="Age", info="Choose between 2 and 100"),
                            gr.Radio(["1", "0", ], label="Anaemia", info="Does person has Anaemia?"),
                            gr.Slider(10, 10000, value=500, label="Creatinine Phosphokinase", info="Choose between 10 and 10000"),
                            gr.Radio(["1", "0", ], label="Diabetes", info="Does person has Diabetes?"),
                            gr.Slider(5, 100, value=500, label="Ejection Fraction", info="Choose between 5 and 500"),
                            gr.Radio(["1", "0", ], label="High Blood Pressure", info="Does person has high blood pressure?"),
                            gr.Slider(50000, 500000, value=500000, label="Platelets", info="Choose between 50,000 and 5,00,000"),
                            gr.Slider(1, 10, value=1, label="Serum Creatinine", info="Choose between 1 and 10"),
                            gr.Slider(50, 500, value=50, label="Serum Sodium", info="Choose between 50 and 500"),
                            gr.Radio(["1", "0", ], label="Sex", info="What is the sex of the person"),
                            gr.Radio(["1", "0", ], label="Smoking", info="Does person smoke?"),
                            gr.Slider(1, 50, value=5, label="Time", info="Choose between 1 and 50"),
                        ],
                         #outputs = gr.outputs.Label(type="auto", label="Answer"),
                     #outputs=[gr.components.Textbox (label ='DeathEvent')],
	                      outputs=["number"],
                         title = title,
                         description = description)

iface.launch(server_name="0.0.0.0",server_port=8001)