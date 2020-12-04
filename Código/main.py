#Librerias necesarias para la implementació
import cv2
import numpy as np
from keras.models import load_model
import pydot
from keras.utils import plot_model
import IPython

#Función para detectar la cara del usuario y determinar una ventana de 200 pixeles de más que el tamáño de la cara
def detect_faces(img, classifier):
    global x, y, h, w
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5) #Detección de las coordenadas
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i        #Guargamos la más grande
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        if x<100 or y<100:
            return None
        else:
            y = y - 100
            x = x - 100
            h = h + 200
            w = w + 200
            frame = gray_frame[y:y + h, x:x + w]         #se determina la ventana
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    return frame


model = load_model('model12.h5')                             #Cargar el modelo
#plot_model(model, to_file="modelo.png",show_shapes=True)    #Plotear la red neuronal
#IPython.display.Image('modelo.png')
model.summary()                                             #Resumen de las capas de la red neuronal
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #Modelo de haarcascade

while True:
    _, Ktest = cap.read()
    K_frame = detect_faces(Ktest, face_cascade)       #Detección del rostro
    if K_frame is not None:
        ven1 = K_frame/255.0                          #Normalización
        K_frame_resize = cv2.resize(ven1, (60, 60), interpolation = cv2.INTER_AREA)  #Ajustar tamaño a la entrada que
                                                                                     #perimte la red
        K_final = np.ndarray(shape=(1,60, 60,1),dtype=float, order='C')
        K_final[0,:,:,0]=K_frame_resize
        Ktest_pred = model.predict(K_final)            #Predicción de las marcas de la cara con el modelo

        for i in range(0, 9, 2):
            Ktest_pred[0,i] = (w * Ktest_pred[0,i])     #Ajuste de los resultados de la preducción al tamaño de la
        for i in range(1,10,2):                         #imagen original
            Ktest_pred[0,i] = (h * Ktest_pred[0,i])

        for i in range(0, 9, 2):   #Dibujo de puntos en la cámara
            cv2.circle(Ktest, (x + int(Ktest_pred[0,i]), y + int(Ktest_pred[0,i+1])), 3, (0, 0, 255), 3)
    cv2.imshow('image', Ktest)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()