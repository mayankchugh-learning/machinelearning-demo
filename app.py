import os
import uuid
import joblib
import json

import gradio as gr
import pandas as pd

from huggingface_hub import CommitScheduler
from pathlib import Path


machine_failure_predictor = joblib.load('model.joblib')

air_temperature_input = gr.Number(label='Air temperature [K]')
process_temperature_input = gr.Number(label='Process temperature [K]')
rotational_speed_input = gr.Number(label='Rotational speed [rpm]')
torque_input = gr.Number(label='Torque [Nm]')
tool_wear_input = gr.Number(label='Tool wear [min]')
type_input = gr.Dropdown(
    ['L', 'M', 'H'],
    label='Type'
)

model_output = gr.Label(label="Machine failure")

def predict_machine_failure(air_temperature, process_temperature, rotational_speed, torque, tool_wear, type):
    sample = {
        'Air temperature [K]': air_temperature,
        'Process temperature [K]': process_temperature,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type': type
    }
    data_point = pd.DataFrame([sample])
    prediction = machine_failure_predictor.predict(data_point).tolist()
            
    return prediction[0]

demo = gr.Interface(
    fn=predict_machine_failure,
    inputs=[air_temperature_input, process_temperature_input, rotational_speed_input, 
            torque_input, tool_wear_input, type_input],
    outputs=model_output,
    title="Machine Failure Predictor",
    description="This API allows you to predict the machine failure status of an equipment",
    allow_flagging="auto",
    concurrency_limit=16
)

demo.queue()
demo.launch(share=False)