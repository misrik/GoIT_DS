import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

# Завантаження збережених моделей
@st.cache_resource
def load_trained_model(model_name):
    return load_model(model_name)

# Мапа класів Fashion MNIST
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Інтерфейс Streamlit
st.title("Fashion MNIST Image Classification")

# Вибір моделі
st.sidebar.header("Select Model")
model_choice = st.sidebar.radio("Choose a model:", ('CNN', 'VGG16'))

# Завантаження вибраної моделі
if model_choice == 'CNN':
    model = load_trained_model('cnn_model.h5')
    history_path = 'cnn_history.npy'
else:
    model = load_trained_model('vgg16_model.h5')
    history_path = 'vgg16_history.npy'

# Завантаження зображення
uploaded_file = st.file_uploader("Upload an image for classification", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Обробка зображення перед передбаченням
    if model_choice == 'CNN':
        image = ImageOps.grayscale(image)  # CNN працює тільки з ч/б зображеннями
        image = image.resize((28, 28))
        image = np.array(image) / 255.0

        # Переконуємось, що у нас правильний розмір
        if image.shape != (28, 28):
            st.error(f"Invalid image shape: {image.shape}. Expected (28, 28)")
        else:
            image = image.reshape((1, 28, 28, 1))  # Остаточна форма
    
    else:  # VGG16
        image = image.resize((48, 48))  # Зміна розміру
        image = np.array(image)  # Конвертація у масив

        # Якщо зображення чорно-біле, додаємо 3 канали
        if len(image.shape) == 2:  
            image = np.stack((image,)*3, axis=-1)

        image = image / 255.0  # Нормалізація
        image = image.reshape((1, 48, 48, 3))  # Остаточна форма
    
    # Передбачення класу
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels.get(predicted_class, "Unknown class")
    
    # Відображення результатів
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_label}")
    
    # Відображення ймовірностей передбачень у таблиці
    st.write("### Confidence Scores:")
    confidence_scores = {class_labels[i]: f"{predictions[0][i]:.4f}" for i in range(10)}
    st.table(confidence_scores)
    
    # Завантаження історії навчання з .npy
    st.subheader("Training Performance")
    try:
        history = np.load(history_path, allow_pickle=True).item()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        ax[0].plot(history['accuracy'], label='Training Accuracy')
        ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        
        ax[1].plot(history['loss'], label='Training Loss')
        ax[1].plot(history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Training history is missing. Ensure history is saved in a .npy file when training the model.")
        
    # Завантаження обробленого зображення
    if model_choice == 'CNN':
        processed_image = image[0].squeeze() * 255  # Видалити зайвий вимір
    else:
        processed_image = image[0] * 255  # Для VGG16 залишаємо 3 канали

    image = Image.fromarray(processed_image.astype(np.uint8))
    
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Download Processed Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )
