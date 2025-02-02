import streamlit as st
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("vae.onnx", providers=["CPUExecutionProvider"])

# Title for the main page
st.title("VAE Visualizer")

# Create an empty list to store slider values
slider_values = []
slider_names = ["Bottom Loop", "", "Line width", "", "", "", "", "", "", "", "", "", "", "", "", "", ]
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
out = np.clip(session.run(None, {"input": slider_array})[0].squeeze(), 0., 1.)
st.image(out, width=300, caption="Generated Image")
