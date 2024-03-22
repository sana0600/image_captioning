from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet')
# Remove the last layer (classification layer) to obtain features
vgg16 = Model(inputs=vgg16.input, outputs=vgg16.layers[-2].output)


caption_generator_model = load_model("models\image_caption_model.h5")

def generate_caption(image_path, vgg_model, caption_model, tokenizer, max_length):
    # Extract features using VGG16
    features = extract_features(image_path)

    # Initialize caption input with start token
    in_text = 'startseq'

    # Iterate to generate each word in the caption
    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')  # Add padding

        # Predict the next word using the caption generation model
        yhat = caption_model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)

        # Map the predicted index to a word
        word = idx_to_word(yhat, tokenizer)

          # Append the predicted word to the input sequence for the next iteration
        in_text += ' ' + word

        # Stop if we reach the end token
        if word is None or word == 'endseq':
            break

    image = Image.open(image_path)
    #plt.imshow(image)
    return in_text

def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = vgg16.predict(img)
    return features

with open(r"models\tokenizer.pkl", 'rb') as file:
    # Rest of your code
    tokenizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("index.html", result="No file provided")

        file = request.files["file"]

        # If the user does not select a file, the browser may submit an empty part without a filename
        if file.filename == "":
            return render_template("index.html", result="No file selected")
        
        file_path = os.path.join(r"static\uploads", file.filename)
        file.save(file_path)
        print("Uploaded file:", file.filename, file_path)
        str = "Uploaded file : " + file.filename
        generated_caption = generate_caption(file_path, vgg16, caption_generator_model, tokenizer, 35)
       
        return render_template("index.html", result=str, generated_caption=generated_caption, image_path = file.filename)


    # If it's a GET request or no file has been submitted yet
    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug = True, port = 5001)
