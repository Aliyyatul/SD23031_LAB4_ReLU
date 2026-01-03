import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", layout="wide")

st.title("ReLU Activation Function")

st.write("ReLU is defined as f(x) = max(0, x)")

# Sidebar sliders (ADJUSTABLE)
st.sidebar.header("Input Range Settings")
x_min = st.sidebar.slider("Minimum x", -20, -5, -10)
x_max = st.sidebar.slider("Maximum x", 5, 20, 10)

# Generate input
x = np.linspace(x_min, x_max, 400)
y = np.maximum(0, x)

# Plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.set_title("ReLU Activation Function")
ax.grid(True)

st.pyplot(fig)

st.write("""
ReLU outputs zero for negative values and linear output for positive values.
It is widely used in hidden layers due to its efficiency.
""")

