import streamlit as st
from PIL import Image
from Pipeline.Predict import predict, load_model_and_labels


model, labels = load_model_and_labels()

st.title("Plant Species Classifier")

st.write("This web app uses a Convolutional Neural Network (CNN) to classify plants based on their images.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

fixed_img_size = (224, 224)

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img.resize(fixed_img_size), caption='Uploaded Image', use_column_width=False, width=224)
    
    
    if st.button("Classify"):

        result = predict(model, img, labels)
        
        st.success(f"The plant species is: **{result}**")
else:
    st.info("Please upload an image to classify.")


st.sidebar.markdown("---")

st.sidebar.write(
    """
    Streamlit CNN model created by [Harishwar](https://www.linkedin.com/in/harishwartg/).

    For more details, please visit the [GitHub repository](https://github.com/HarishwarTG/CNN_Plant_Classification).

    **Note:** This is a personal learning project, and the CNN model was trained with limited hardware resources, which may result in lower accuracy.
    """
)