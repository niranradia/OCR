import predict
import draw

while True:
    try:
        draw.draw()
        predict.predict_image("drawing.png",10)
        input()
    except:
        input()
        exit()