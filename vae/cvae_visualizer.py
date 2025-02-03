import streamlit as st
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("cvae.onnx", providers=["CPUExecutionProvider"])

# Title for the main page
st.title("VAE Visualizer")

# Create an empty list to store slider values
slider_values = []
slider_names = [
    "Bottom Loop",
    "",
    "Line width",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]
digit = st.sidebar.number_input(
    "Enter the digit to generate",
    min_value=0,
    max_value=9,
    value=0,
    key="digit",
)
# Generate 16 sliders in the sidebar
for i in range(16):
    value = st.sidebar.slider(
        label=slider_names[i],
        key=f"slider_{i}",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,  # Default value
        step=0.01,
    )
    slider_values.append(value)

# Convert the list of slider values to a NumPy array
slider_array = np.array(slider_values).reshape(1, 16).astype(np.float32)
st.write(f"Digit is {digit}")
st.write(np.array([digit]))
out = np.clip(
    session.run(
        None,
        {
            "input": slider_array,
            "label": np.array([digit or 0.0]).astype(np.int64),
        },
    )[0].squeeze(),
    0.0,
    1.0,
)
st.image(out, width=300, caption="Generated Image")
