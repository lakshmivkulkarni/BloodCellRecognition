
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Blood Cell AI",
    page_icon="🩸",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "models" / "blood_cell_model_v3.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "Basophil", "Eosinophil", "Lymphocyte",
    "Monocyte", "Neutrophil"
]

st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}
.main-title {
    color: #12345a;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    color: #52657a;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">🩸 Blood Cell Recognition System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-powered White Blood Cell Classification</p>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Upload Your Blood Cell Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Blood Cell Image",
            use_container_width=True
        )

    with col2:
        st.markdown("### Image Analysis")
        st.write("Image uploaded successfully.")
        st.write("Model: MobileNetV2 (V3)")
        st.write("Classes: 5 White Blood Cell Types")

        if st.button("🔍 Predict Blood Cell", use_container_width=True):

            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.success("Prediction Completed!")

            st.markdown("### Prediction Result")
            st.subheader(predicted_class)
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.markdown("### Class Probabilities")

            probability_data = pd.DataFrame({
                "Cell Type": class_names,
                "Probability (%)": predictions * 100
            }).set_index("Cell Type")

            st.bar_chart(probability_data)

            st.info(
                "This is an educational image-classification "
                "prototype, not a medical diagnostic tool."
            )

st.divider()

st.subheader("About the Project")

st.write("""
The Blood Cell Recognition System uses deep learning
and the MobileNetV2 architecture to classify white
blood cell images into five categories.
""")

st.markdown("""
**Technologies Used**
- Python
- TensorFlow and Keras
- MobileNetV2
- Streamlit
- NumPy and Pandas
""")

st.caption("Blood Cell AI | Academic Project")
