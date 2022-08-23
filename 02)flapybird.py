#container generic detect hands in the mediapipe
import mediapipe as mp
import cv2
import keyboard

# inicializamos la clase Hands y almacenarla en una variable
poseMp = mp.solutions.pose
# cargamos componente con las herramientas que nos permitira dibujar mas adelante
drawingMp = mp.solutions.drawing_utils
# cargamos los estilos en la variable mp_drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles
# iniciamos una captura de video en la camara 1
cap = cv2.VideoCapture(1)
# save height and width image
height, width = [0, 0]

global flystate

flystate = True

# ejecutamos del bloque de deteccion
def drawLineFly(posLine1Y, posLine2Y, image):
    cv2.line(image,(0,posLine1Y),(width,posLine2Y),(255,0,0),4)


def fly(posAla1, posAla2, image, posLine1Y, posLine2Y):
    global flystate
    if posLine1Y>posAla1 and posLine2Y>posAla2 and flystate:
        print("vuela")
        flystate = False
        keyboard.press_and_release("space")
    else:
        if posLine1Y < posAla1 and posLine2Y < posAla2 and flystate == False:
            print("no vuela")
            flystate = True

def flappy(pose_landmarks, image):
    posLine1Y = int(pose_landmarks.landmark[poseMp.PoseLandmark.RIGHT_SHOULDER].y*height)
    posLine2Y = int(pose_landmarks.landmark[poseMp.PoseLandmark.LEFT_SHOULDER].y*height)
    drawLineFly(posLine1Y,posLine2Y,image)
    posAla1 = int(pose_landmarks.landmark[poseMp.PoseLandmark.RIGHT_INDEX].y*height)
    posAla2 = int(pose_landmarks.landmark[poseMp.PoseLandmark.LEFT_INDEX].y * height)
    fly(posAla1,posAla2,image,posLine1Y,posLine2Y)

with poseMp.Pose(static_image_mode=False,
                   min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) as pose:
    # mientras la camara este en ejecucion
    while cap.isOpened():
        # guardamos en la variable succes el estado de la captura y en image la captura
        success, image = cap.read()
        if not success:
            print("camara vacia")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # get shape image
        if height == 0:
            # return height, width and channels
            height, width, _ = image.shape
            print(height, width)
        # convertimos la imagen de bgr a rgb debido a que la funcion hands process acepta rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Procesa una imagen RGB y devuelve los puntos de referencia de la mano y la destreza de cada mano detectada
        results = pose.process(image)
        # convertimos la imagen rgb a bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        drawingMp.draw_landmarks(
            image,
            results.pose_landmarks,
            poseMp.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #*************************************
        flappy(results.pose_landmarks,image)
        #*************************************

        # Voltee la imagen horizontalmente para obtener una vista de selfie.
        cv2.imshow('MediaPipeJHR', cv2.flip(image, 1))
        # en caso de teclear la letra q suspendemos la operacion
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Cierra el archivo de video o el dispositivo de captura.
cap.release()
