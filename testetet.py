import streamlit as st
import joblib
import numpy as np
import c2v

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
 
# Ladda in MNIST-datasetet
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)
X_train = X[:60000]
y_train = y[:60000]
 
# Träna en KNN-klassificerare
knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=3, weights='distance', metric='euclidean')
knn_clf.fit(X, y)

# Funktion för att göra förutsägelser
def predict_digit(image):
    # Förbered bilden för modellen
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray_image.flatten().reshape(1, -1)
    
    # Gör förutsägelser med modellen
    prediction = knn_clf.predict(flattened_image)
    return prediction[0]

# Streamlit UI
st.title("Handwritten Digit Recognition App")
st.write("Use your camera to capture a handwritten digit and let the model predict it.")

# Öppna kameran
cap = cv2.VideoCapture(0)

# Visa livevideo och göra förutsägelser
while True:
    ret, frame = cap.read()  # Läs in en bild från kameran
    if ret:
        digit = predict_digit(frame)  # Gör förutsägelse
        st.image(frame, channels="BGR", use_column_width=True, caption=f"The predicted digit is: {digit}")
        if st.button("Take Picture"):
            # Visa bilden som togs
            st.image(frame, channels="BGR", use_column_width=True, caption="Picture Taken")
            # Gör en förutsägelse på den tagna bilden
            digit = predict_digit(frame)
            st.write(f"The predicted digit is: {digit}")
    else:
        st.error("Error: Unable to access camera.")
        break

# Stäng kameran när användaren avslutar appen
cap.release()
