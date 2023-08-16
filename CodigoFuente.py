# -*- coding: utf8 -*-

#Importamos las librerías necesarias
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import skfuzzy as fuzz

dataPath = 'E:/Investigaciones/Deteccion Temperatura Imagenes Termicas/Imagenes_Termicas'
peopleList = os.listdir(dataPath)
#print('Lista de personas: ', peopleList)

#Cargamos una fuente de texto
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
archivo = open('Datos.csv','w')

for nameDir in peopleList:
    
    personPath = dataPath + '/' + nameDir    

    #Abrimos la imagen
    imagen_termica = Image.open(personPath)

    plt.imshow(imagen_termica)

    #Transformamos la foto a una lista que contiene los datos RGB
    lista_imagen = list(imagen_termica.getdata())

    print(lista_imagen)

    #Transformamos los datos de la lista a escala de grises RANGO [0 - 255]
    grises = [round((0.2125 * lista_imagen[x][0]) + (0.7154 * lista_imagen[x][1]) + (0.072 * lista_imagen[x][2])) for x in range(len(lista_imagen))]

    print(grises)

    #Creamos una nueva imagen con el mismo tamaño de la imagen inicial
    imagen_grises = Image.new('L', imagen_termica.size)

    plt.imshow(imagen_grises)

    #Dentro de la nueva imagen ponemos los datos obtenidos al calcular
    #la escala de grises
    imagen_grises.putdata(grises)

    plt.imshow(imagen_grises)

    #Volvemos a transformar a una lista para obtener el maximo valor de la escala de grises
    lista_imagen_grises = list(imagen_grises.getdata())
    maximo = max(lista_imagen_grises)
    minimo = min(lista_imagen_grises)

    print(lista_imagen_grises)

    print("Máximo valor de la escala de grises: "+str(maximo))
    print("Mínimo valor de la escala de grises: "+str(minimo))

    #Ponemos el valor maximo como un array con la funcion NumPy para que lo reconozca
    #la funcion fuzz como un valor array
    escala_gris_pixel = np.arange(maximo,maximo + 0.1,0.1)

    print(escala_gris_pixel)

    pixel= np.arange(0,256,1)

    print(pixel)

    temperatura = np.arange(30, 45.1, 0.1)

    print(temperatura)

    pixel1 = fuzz.trapmf(pixel, [0,0,20,40])
    pixel2 = fuzz.trimf(pixel, [30,40,50])
    pixel3 = fuzz.trimf(pixel, [40,50,60])
    pixel4 = fuzz.trimf(pixel, [50,60,70])
    pixel5 = fuzz.trimf(pixel, [60,70,80])
    pixel6 = fuzz.trimf(pixel, [70,80,90])
    pixel7 = fuzz.trimf(pixel, [80,90,100])
    pixel8 = fuzz.trimf(pixel, [90,100,110])
    pixel9 = fuzz.trimf(pixel, [100,110,120])
    pixel10 = fuzz.trimf(pixel, [110,120,130])
    pixel11 = fuzz.trimf(pixel, [120,130,140])
    pixel12 = fuzz.trimf(pixel, [130,140,150])
    pixel13 = fuzz.trimf(pixel, [140,150,160])
    pixel14 = fuzz.trimf(pixel, [150,160,170])
    pixel15 = fuzz.trimf(pixel, [160,170,180])
    pixel16 = fuzz.trimf(pixel, [170,180,190])
    pixel17 = fuzz.trimf(pixel, [180,190,200])
    pixel18 = fuzz.trimf(pixel, [190,200,210])
    pixel19 = fuzz.trimf(pixel, [200,210,220])
    pixel20 = fuzz.trimf(pixel, [210,220,230])
    pixel21 = fuzz.trimf(pixel, [220,230,232])
    pixel22 = fuzz.trimf(pixel, [230,232,234])
    pixel23 = fuzz.trimf(pixel, [232,234,236])
    pixel24 = fuzz.trimf(pixel, [234,236,238])
    pixel25 = fuzz.trimf(pixel, [236,238,240])
    pixel26 = fuzz.trimf(pixel, [238,240,242])
    pixel27 = fuzz.trimf(pixel, [240,242,246])
    pixel28 = fuzz.trimf(pixel, [244,246,248])
    pixel29 = fuzz.trimf(pixel, [246,248,250])
    pixel30 = fuzz.trimf(pixel, [248,250,252])
    pixel31 = fuzz.trimf(pixel, [250,252,254])
    pixel32 = fuzz.trapmf(pixel, [252,254,255,255])


    plt.plot(pixel, pixel1, label="1")
    plt.plot(pixel, pixel2, label="2")
    plt.plot(pixel, pixel3, label="3")
    plt.plot(pixel, pixel4, label="4")
    plt.plot(pixel, pixel5, label="5")
    plt.plot(pixel, pixel6, label="6")
    plt.plot(pixel, pixel7, label="7")
    plt.plot(pixel, pixel8, label="8")
    plt.plot(pixel, pixel9, label="9")
    plt.plot(pixel, pixel10, label="10")
    plt.plot(pixel, pixel11, label="11")
    plt.plot(pixel, pixel12, label="12")
    plt.plot(pixel, pixel13, label="13")
    plt.plot(pixel, pixel14, label="14")
    plt.plot(pixel, pixel15, label="15")
    plt.plot(pixel, pixel16, label="16")
    plt.plot(pixel, pixel17, label="17")
    plt.plot(pixel, pixel18, label="18")
    plt.plot(pixel, pixel19, label="19")
    plt.plot(pixel, pixel20, label="20")
    plt.plot(pixel, pixel21, label="21")
    plt.plot(pixel, pixel22, label="22")
    plt.plot(pixel, pixel23, label="23")
    plt.plot(pixel, pixel24, label="24")
    plt.plot(pixel, pixel25, label="25")
    plt.plot(pixel, pixel26, label="26")
    plt.plot(pixel, pixel27, label="27")
    plt.plot(pixel, pixel28, label="28")
    plt.plot(pixel, pixel29, label="29")
    plt.plot(pixel, pixel30, label="30")
    plt.plot(pixel, pixel31, label="31")
    plt.plot(pixel, pixel32, label="32")

    plt.xlabel('Intensidad de luz emitida por un pixel')
    plt.show()

    temperatura1  = fuzz.trapmf(temperatura, [30,30,31,32])
    temperatura2  = fuzz.trimf(temperatura, [31,32,33])
    temperatura3  = fuzz.trimf(temperatura, [32,33,33.1])
    temperatura4  = fuzz.trimf(temperatura, [33,33.1,33.2])
    temperatura5  = fuzz.trimf(temperatura, [33.1,33.2,33.3])
    temperatura6  = fuzz.trimf(temperatura, [33.2,33.3,33.4])
    temperatura7  = fuzz.trimf(temperatura, [33.3,33.4,33.5])
    temperatura8  = fuzz.trimf(temperatura, [33.4,33.5,33.6])
    temperatura9  = fuzz.trimf(temperatura, [33.5,33.6,33.7])
    temperatura10  = fuzz.trimf(temperatura, [33.6,33.7,33.8])
    temperatura11  = fuzz.trimf(temperatura, [33.7,33.8,33.9])
    temperatura12  = fuzz.trimf(temperatura, [33.8,33.9,34])
    temperatura13  = fuzz.trimf(temperatura, [33.9,34,34.1])
    temperatura14  = fuzz.trimf(temperatura, [34,34.1,34.2])
    temperatura15  = fuzz.trimf(temperatura, [34.1,34.2,34.3])
    temperatura16  = fuzz.trimf(temperatura, [34.2,34.3,34.4])
    temperatura17  = fuzz.trimf(temperatura, [34.3,34.4,34.5])
    temperatura18  = fuzz.trimf(temperatura, [34.4,34.5,34.6])
    temperatura19  = fuzz.trimf(temperatura, [34.5,34.6,34.7])
    temperatura20  = fuzz.trimf(temperatura, [34.6,34.7,34.8])
    temperatura21  = fuzz.trimf(temperatura, [34.7,34.8,34.9])
    temperatura22  = fuzz.trimf(temperatura, [34.8,34.9,35])
    temperatura23  = fuzz.trimf(temperatura, [33.9,34,34.2])
    temperatura24  = fuzz.trimf(temperatura, [34,34.2,34.6])
    temperatura25  = fuzz.trimf(temperatura, [34.2,34.6,34.8])
    temperatura26  = fuzz.trimf(temperatura, [34,34.5,35])
    temperatura27  = fuzz.trimf(temperatura, [34.5,35,35.5])
    temperatura28  = fuzz.trimf(temperatura, [34,35,35.5])
    temperatura29  = fuzz.trimf(temperatura, [35,36,36.5])
    temperatura30  = fuzz.trimf(temperatura, [36,36.5,37])
    temperatura31  = fuzz.trimf(temperatura, [36.5,37,40])
    temperatura32  = fuzz.trapmf(temperatura, [38,39,40,40])

    plt.plot(temperatura, temperatura1, label="1")
    plt.plot(temperatura, temperatura2, label="2")
    plt.plot(temperatura, temperatura3, label="3")
    plt.plot(temperatura, temperatura4, label="4")
    plt.plot(temperatura, temperatura5, label="5")
    plt.plot(temperatura, temperatura6, label="6")
    plt.plot(temperatura, temperatura7, label="7")
    plt.plot(temperatura, temperatura8, label="8")
    plt.plot(temperatura, temperatura9, label="9")
    plt.plot(temperatura, temperatura10, label="10")
    plt.plot(temperatura, temperatura11, label="11")
    plt.plot(temperatura, temperatura12, label="12")
    plt.plot(temperatura, temperatura13, label="13")
    plt.plot(temperatura, temperatura14, label="14")
    plt.plot(temperatura, temperatura15, label="15")
    plt.plot(temperatura, temperatura16, label="16")
    plt.plot(temperatura, temperatura17, label="17")
    plt.plot(temperatura, temperatura18, label="18")
    plt.plot(temperatura, temperatura19, label="19")
    plt.plot(temperatura, temperatura20, label="20")
    plt.plot(temperatura, temperatura21, label="21")
    plt.plot(temperatura, temperatura22, label="22")
    plt.plot(temperatura, temperatura23, label="23")
    plt.plot(temperatura, temperatura24, label="24")
    plt.plot(temperatura, temperatura25, label="25")
    plt.plot(temperatura, temperatura26, label="26")
    plt.plot(temperatura, temperatura27, label="27")
    plt.plot(temperatura, temperatura28, label="28")
    plt.plot(temperatura, temperatura29, label="29")
    plt.plot(temperatura, temperatura30, label="30")
    plt.plot(temperatura, temperatura31, label="31")
    plt.plot(temperatura, temperatura32, label="32")

    plt.xlabel('TEMPERATURA')
    plt.show()

    valor_pixel1 = fuzz.trapmf(escala_gris_pixel, [0,0,8,16])
    valor_pixel2 = fuzz.trimf(escala_gris_pixel, [8,16,24])
    valor_pixel3 = fuzz.trimf(escala_gris_pixel, [16,24,32])
    valor_pixel4 = fuzz.trimf(escala_gris_pixel, [24,32,40])
    valor_pixel5 = fuzz.trimf(escala_gris_pixel, [32,40,48])
    valor_pixel6 = fuzz.trimf(escala_gris_pixel, [40,48,56])
    valor_pixel7 = fuzz.trimf(escala_gris_pixel, [48,56,64])
    valor_pixel8 = fuzz.trimf(escala_gris_pixel, [56,64,72])
    valor_pixel9 = fuzz.trimf(escala_gris_pixel, [64,72,80])
    valor_pixel10 = fuzz.trimf(escala_gris_pixel, [80,88,96])
    valor_pixel11 = fuzz.trimf(escala_gris_pixel, [88,96,104])
    valor_pixel12 = fuzz.trimf(escala_gris_pixel, [96,104,112])
    valor_pixel13 = fuzz.trimf(escala_gris_pixel, [104,112,120])
    valor_pixel14 = fuzz.trimf(escala_gris_pixel, [112,120,128])
    valor_pixel15 = fuzz.trimf(escala_gris_pixel, [120,128,136])
    valor_pixel16 = fuzz.trimf(escala_gris_pixel, [128,136,142])
    valor_pixel17 = fuzz.trimf(escala_gris_pixel, [136,142,150])
    valor_pixel18 = fuzz.trimf(escala_gris_pixel, [142,150,158])
    valor_pixel19 = fuzz.trimf(escala_gris_pixel, [150,158,164])
    valor_pixel20 = fuzz.trimf(escala_gris_pixel, [158,164,172])
    valor_pixel21 = fuzz.trimf(escala_gris_pixel, [164,172,180])
    valor_pixel22 = fuzz.trimf(escala_gris_pixel, [172,180,188])
    valor_pixel23 = fuzz.trimf(escala_gris_pixel, [180,188,196])
    valor_pixel24 = fuzz.trimf(escala_gris_pixel, [188,196,204])
    valor_pixel25 = fuzz.trimf(escala_gris_pixel, [196,204,212])
    valor_pixel26 = fuzz.trimf(escala_gris_pixel, [204,212,220])
    valor_pixel27 = fuzz.trimf(escala_gris_pixel, [212,220,228])
    valor_pixel28 = fuzz.trimf(escala_gris_pixel, [220,228,236])
    valor_pixel29 = fuzz.trimf(escala_gris_pixel, [228,236,244])
    valor_pixel30 = fuzz.trimf(escala_gris_pixel, [236,244,252])
    valor_pixel31 = fuzz.trimf(escala_gris_pixel, [244,252,255])
    valor_pixel32 = fuzz.trapmf(escala_gris_pixel, [252,253,255,255])


    plt.plot(pixel, pixel1, label="1")
    plt.plot(pixel, pixel2, label="2")
    plt.plot(pixel, pixel3, label="3")
    plt.plot(pixel, pixel4, label="4")
    plt.plot(pixel, pixel5, label="5")
    plt.plot(pixel, pixel6, label="6")
    plt.plot(pixel, pixel7, label="7")
    plt.plot(pixel, pixel8, label="8")
    plt.plot(pixel, pixel9, label="9")
    plt.plot(pixel, pixel10, label="10")
    plt.plot(pixel, pixel11, label="11")
    plt.plot(pixel, pixel12, label="12")
    plt.plot(pixel, pixel13, label="13")
    plt.plot(pixel, pixel14, label="14")
    plt.plot(pixel, pixel15, label="15")
    plt.plot(pixel, pixel16, label="16")
    plt.plot(pixel, pixel17, label="17")
    plt.plot(pixel, pixel18, label="18")
    plt.plot(pixel, pixel19, label="19")
    plt.plot(pixel, pixel20, label="20")
    plt.plot(pixel, pixel21, label="21")
    plt.plot(pixel, pixel22, label="22")
    plt.plot(pixel, pixel23, label="23")
    plt.plot(pixel, pixel24, label="24")
    plt.plot(pixel, pixel25, label="25")
    plt.plot(pixel, pixel26, label="26")
    plt.plot(pixel, pixel27, label="27")
    plt.plot(pixel, pixel28, label="28")
    plt.plot(pixel, pixel29, label="29")
    plt.plot(pixel, pixel30, label="30")
    plt.plot(pixel, pixel31, label="31")
    plt.plot(pixel, pixel32, label="32")
    plt.plot([escala_gris_pixel, escala_gris_pixel], [0.0, 1.0], linestyle="--")
    plt.plot(escala_gris_pixel, len(valor_pixel1), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel2), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel3), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel4), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel5), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel6), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel7), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel8), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel9), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel10), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel11), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel12), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel13), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel14), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel15), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel16), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel17), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel18), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel19), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel20), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel21), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel22), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel23), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel24), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel25), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel26), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel27), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel28), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel29), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel30), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel31), 'x')
    plt.plot(escala_gris_pixel, len(valor_pixel32), 'x')


    plt.xlabel('VALOR MAXIMO DE LA ESCALA DE GRISES')
    plt.show()

    print('valor_pixel1 = ',valor_pixel1)
    print('valor_pixel2 = ',valor_pixel2)
    print('valor_pixel3 = ',valor_pixel3)
    print('valor_pixel4 = ',valor_pixel4)
    print('valor_pixel5 = ',valor_pixel5)
    print('valor_pixel6 = ',valor_pixel6)
    print('valor_pixel7 = ',valor_pixel7)
    print('valor_pixel8 = ',valor_pixel8)
    print('valor_pixel9 = ',valor_pixel9)
    print('valor_pixel10 = ',valor_pixel10)
    print('valor_pixel11 = ',valor_pixel11)
    print('valor_pixel12 = ',valor_pixel12)
    print('valor_pixel13 = ',valor_pixel13)
    print('valor_pixel14 = ',valor_pixel14)
    print('valor_pixel15 = ',valor_pixel15)
    print('valor_pixel16 = ',valor_pixel16)
    print('valor_pixel17 = ',valor_pixel17)
    print('valor_pixel18 = ',valor_pixel18)
    print('valor_pixel19 = ',valor_pixel19)
    print('valor_pixel20 = ',valor_pixel20)
    print('valor_pixel21 = ',valor_pixel21)
    print('valor_pixel22 = ',valor_pixel22)
    print('valor_pixel23 = ',valor_pixel23)
    print('valor_pixel24 = ',valor_pixel24)
    print('valor_pixel25 = ',valor_pixel25)
    print('valor_pixel26 = ',valor_pixel26)
    print('valor_pixel27 = ',valor_pixel27)
    print('valor_pixel28 = ',valor_pixel28)
    print('valor_pixel29 = ',valor_pixel29)
    print('valor_pixel30 = ',valor_pixel30)
    print('valor_pixel31 = ',valor_pixel31)
    print('valor_pixel32 = ',valor_pixel32)


    def cortar(valor_entrada, funcion_salida):
        valor_entrada = float(valor_entrada)
        vector_puntos_cortes = np.zeros(funcion_salida.size)
        if (type(valor_entrada) is int) or (type(valor_entrada) is float):
            for i in range(funcion_salida.size):
                vector_puntos_cortes[i] = min(valor_entrada, funcion_salida[i])
            return vector_puntos_cortes
        else:
            return -1

    temp1p = cortar(valor_pixel1,temperatura1)
    temp2p = cortar(valor_pixel2,temperatura2)
    temp3p = cortar(valor_pixel3,temperatura3)
    temp4p = cortar(valor_pixel4,temperatura4)
    temp5p = cortar(valor_pixel5,temperatura5)
    temp6p = cortar(valor_pixel6,temperatura6)
    temp7p = cortar(valor_pixel7,temperatura7)
    temp8p = cortar(valor_pixel8,temperatura8)
    temp9p = cortar(valor_pixel9,temperatura9)
    temp10p = cortar(valor_pixel10,temperatura10)
    temp11p = cortar(valor_pixel11,temperatura11)
    temp12p = cortar(valor_pixel12,temperatura12)
    temp13p = cortar(valor_pixel13,temperatura13)
    temp14p = cortar(valor_pixel14,temperatura14)
    temp15p = cortar(valor_pixel15,temperatura15)
    temp16p = cortar(valor_pixel16,temperatura16)
    temp17p = cortar(valor_pixel17,temperatura17)
    temp18p = cortar(valor_pixel18,temperatura18)
    temp19p = cortar(valor_pixel19,temperatura19)
    temp20p = cortar(valor_pixel20,temperatura20)
    temp21p = cortar(valor_pixel21,temperatura21)
    temp22p = cortar(valor_pixel22,temperatura22)
    temp23p = cortar(valor_pixel23,temperatura23)
    temp24p = cortar(valor_pixel24,temperatura24)
    temp25p = cortar(valor_pixel25,temperatura25)
    temp26p = cortar(valor_pixel26,temperatura26)
    temp27p = cortar(valor_pixel27,temperatura27)
    temp28p = cortar(valor_pixel28,temperatura28)
    temp29p = cortar(valor_pixel29,temperatura29)
    temp30p = cortar(valor_pixel30,temperatura30)
    temp31p = cortar(valor_pixel31,temperatura31)
    temp32p = cortar(valor_pixel32,temperatura32)

    def union(arrays_de_entrada):
        array_unido = np.zeros(arrays_de_entrada[0].size)
        for j in range(len(arrays_de_entrada)):
            for i in range(array_unido.size):
                array_unido[i] = max(array_unido[i], arrays_de_entrada[j][i])
        return array_unido

    tempp = union([temp1p,temp2p,temp3p,temp4p,temp5p,temp6p,temp7p,
                   temp8p,temp9p,temp10p,temp11p,temp12p,temp13p,temp14p,
                   temp15p,temp16p,temp17p,temp18p,temp19p,temp20p,temp21p,
                   temp22p,temp23p,temp24p,temp25p,temp26p,temp27p,temp28p,
                   temp29p,temp30p,temp31p,temp32p])

    print(tempp)

    plt.plot(temperatura, temperatura1, label="1")
    plt.plot(temperatura, temperatura2, label="2")
    plt.plot(temperatura, temperatura3, label="3")
    plt.plot(temperatura, temperatura4, label="4")
    plt.plot(temperatura, temperatura5, label="5")
    plt.plot(temperatura, temperatura6, label="6")
    plt.plot(temperatura, temperatura7, label="7")
    plt.plot(temperatura, temperatura8, label="8")
    plt.plot(temperatura, temperatura9, label="9")
    plt.plot(temperatura, temperatura10, label="10")
    plt.plot(temperatura, temperatura11, label="11")
    plt.plot(temperatura, temperatura12, label="12")
    plt.plot(temperatura, temperatura13, label="13")
    plt.plot(temperatura, temperatura14, label="14")
    plt.plot(temperatura, temperatura15, label="15")
    plt.plot(temperatura, temperatura16, label="16")
    plt.plot(temperatura, temperatura17, label="17")
    plt.plot(temperatura, temperatura18, label="18")
    plt.plot(temperatura, temperatura19, label="19")
    plt.plot(temperatura, temperatura20, label="20")
    plt.plot(temperatura, temperatura21, label="21")
    plt.plot(temperatura, temperatura22, label="22")
    plt.plot(temperatura, temperatura23, label="23")
    plt.plot(temperatura, temperatura24, label="24")
    plt.plot(temperatura, temperatura25, label="25")
    plt.plot(temperatura, temperatura26, label="26")
    plt.plot(temperatura, temperatura27, label="27")
    plt.plot(temperatura, temperatura28, label="28")
    plt.plot(temperatura, temperatura29, label="29")
    plt.plot(temperatura, temperatura30, label="30")
    plt.plot(temperatura, temperatura31, label="31")
    plt.plot(temperatura, temperatura32, label="32")
    plt.plot(temperatura, tempp, label="Union", linewidth=3)
    plt.show()

    metodo_centroide = fuzz.defuzz(temperatura,tempp,'centroid')
    metodo_bisectriz = fuzz.defuzz(temperatura,tempp,'bisector')
    metodo_media_central = fuzz.defuzz(temperatura,tempp,'MOM')
    metodo_minimo_central = fuzz.defuzz(temperatura,tempp,'SOM')
    metodo_maximo_central = fuzz.defuzz(temperatura,tempp,'LOM')

    print ("metodo_centroide = ",metodo_centroide)
    print ("metodo_bisectriz = ",metodo_bisectriz)
    print ("metodo_media_central = ",metodo_media_central)
    print ("metodo_minimo_central = ",metodo_minimo_central)
    print ("metodo_maximo_central = ",metodo_maximo_central)

    plt.plot(temperatura,tempp)
    plt.plot([metodo_centroide, metodo_centroide],[0,1])
    plt.show()

    if(maximo>0 & maximo<255):
    
        #Transformamos la imagen con numpy para usar la libreria OpenCV para obtener la region de interes
        imagen = np.array(imagen_termica)
        imagen_real = np.array(imagen_termica)
        imagen_gris = np.array(imagen_grises)

        plt.subplot(1, 2, 1)
        plt.imshow(imagen_real)
        plt.subplot(1, 2, 2)
        plt.imshow(imagen_gris)

        img_g = imagen[:,:,1]

        plt.hist(img_g.flatten(), 100, [0,255], color = "b")

        imagen_hsv = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2HSV)

        plt.imshow(imagen_hsv)

        umbral_bajo = np.array([0, 0, (maximo-5)])
        umbral_alto = np.array([0, 0, 255])

        print(umbral_alto)
        print(umbral_bajo)

        fondo = cv2.inRange(imagen_hsv, umbral_bajo, umbral_alto)
        filtro = cv2.bitwise_and(imagen_real, imagen_hsv, mask=fondo)

        plt.subplot(1, 2, 1)
        plt.imshow(fondo)
        plt.subplot(1, 2, 2)
        plt.imshow(filtro)

        puntos_calor = cv2.bitwise_not(fondo)

        plt.imshow(puntos_calor)

        kernel = np.ones((3,3),np.uint8)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_OPEN,kernel)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_CLOSE,kernel)

        print(kernel)

        plt.imshow(puntos_calor)

        contornos,_ = cv2.findContours(puntos_calor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contor=cv2.drawContours(imagen_real, contornos, -1, (0,255,0), 2)

        for i in contornos:
            #Calcular el centro a partir de los momentos
            momentos = cv2.moments(i)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            
            #Dibujar el centro
            centros = cv2.circle(imagen_real,(cx, cy), 3, (0,0,255), -1)
         
            #Escribimos las coordenadas del centro
            #cv2.putText(imagen_real,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
            if(metodo_centroide>=37.5):
                cv2.putText(imagen_real,str(round(metodo_centroide,2)) + " GRADOS FIEBRE", (10,30), font, 0.5,(0,0,255),1)

            elif(metodo_centroide<=35.5):
                cv2.putText(imagen_real,str(round(metodo_centroide,2)) + " GRADOS HIPOTERMIA", (10,30), font, 0.5,(255,0,0),1)
                
            else:
                cv2.putText(imagen_real,str(round(metodo_centroide,2)) + " GRADOS NORMAL" ,(10,30), font, 0.5,(0,255,0),1)

            
        e = nameDir,metodo_centroide,maximo
        print(e)
        archivo.write(str(e)+'\n')
    elif(maximo>=255):
        #Transformamos la imagen con numpy para usar la libreria OpenCV para obtener la region de interes
        imagen = np.array(imagen_termica)
        imagen_real = np.array(imagen_termica)
        imagen_gris = np.array(imagen_grises)

        plt.subplot(1, 2, 1)
        plt.imshow(imagen_real)
        plt.subplot(1, 2, 2)
        plt.imshow(imagen_gris)

        img_g = imagen[:,:,1]

        plt.hist(img_g.flatten(), 100, [0,255], color = "b")

        imagen_hsv = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2HSV)

        plt.imshow(imagen_hsv)

        umbral_bajo = np.array([0, 0, (maximo-5)])
        umbral_alto = np.array([0, 0, 255])

        print(umbral_alto)
        print(umbral_bajo)

        fondo = cv2.inRange(imagen_hsv, umbral_bajo, umbral_alto)
        filtro = cv2.bitwise_and(imagen_real, imagen_hsv, mask=fondo)

        plt.subplot(1, 2, 1)
        plt.imshow(fondo)
        plt.subplot(1, 2, 2)
        plt.imshow(filtro)

        puntos_calor = cv2.bitwise_not(fondo)

        plt.imshow(puntos_calor)

        kernel = np.ones((3,3),np.uint8)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_OPEN,kernel)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_CLOSE,kernel)

        print(kernel)

        plt.imshow(puntos_calor)

        contornos,_ = cv2.findContours(puntos_calor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contor=cv2.drawContours(imagen_real, contornos, -1, (0,255,0), 2)

        for i in contornos:
            #Calcular el centro a partir de los momentos
            momentos = cv2.moments(i)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            
            #Dibujar el centro
            centros = cv2.circle(imagen_real,(cx, cy), 3, (0,0,255), -1)
         
            #Escribimos las coordenadas del centro
            #cv2.putText(imagen_real,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
            cv2.putText(imagen_real,">45 GRADOS FUERA DE RANGO" ,(2,2), font, 0.5,(255,255,255),1)

            
        e = nameDir,">45°C"
        print(e)
        archivo.write(str(e)+'\n')

    elif(maximo<=0):
        #Transformamos la imagen con numpy para usar la libreria OpenCV para obtener la region de interes
        imagen = np.array(imagen_termica)
        imagen_real = np.array(imagen_termica)
        imagen_gris = np.array(imagen_grises)

        plt.subplot(1, 2, 1)
        plt.imshow(imagen_real)
        plt.subplot(1, 2, 2)
        plt.imshow(imagen_gris)

        img_g = imagen[:,:,1]

        plt.hist(img_g.flatten(), 100, [0,255], color = "b")

        imagen_hsv = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2HSV)

        plt.imshow(imagen_hsv)

        umbral_bajo = np.array([0, 0, (maximo-5)])
        umbral_alto = np.array([0, 0, 255])

        print(umbral_alto)
        print(umbral_bajo)

        fondo = cv2.inRange(imagen_hsv, umbral_bajo, umbral_alto)
        filtro = cv2.bitwise_and(imagen_real, imagen_hsv, mask=fondo)

        plt.subplot(1, 2, 1)
        plt.imshow(fondo)
        plt.subplot(1, 2, 2)
        plt.imshow(filtro)

        puntos_calor = cv2.bitwise_not(fondo)

        plt.imshow(puntos_calor)

        kernel = np.ones((3,3),np.uint8)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_OPEN,kernel)
        puntos_calor = cv2.morphologyEx(puntos_calor,cv2.MORPH_CLOSE,kernel)

        print(kernel)

        plt.imshow(puntos_calor)

        contornos,_ = cv2.findContours(puntos_calor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contor=cv2.drawContours(imagen_real, contornos, -1, (0,255,0), 2)

        for i in contornos:
            #Calcular el centro a partir de los momentos
            momentos = cv2.moments(i)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            
            #Dibujar el centro
            centros = cv2.circle(imagen_real,(cx, cy), 3, (0,0,255), -1)
         
            #Escribimos las coordenadas del centro
            #cv2.putText(imagen_real,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
            cv2.putText(imagen_real,"<30 GRADOS FUERA DE RANGO" ,(2,2), font, 0.5,(255,255,255),1)

            
        e = nameDir,"<30°C"
        print(e)
        archivo.write(str(e)+'\n')
    
    #Mostramos la imagen inicial
    plt.subplot(2, 3, 1)
    plt.imshow(imagen, cmap=plt.cm.gray)

    #Mostramos la imagen filtrada
    plt.subplot(2, 3, 2)
    plt.imshow(fondo)

    #Mostramos la imagen con los puntos de interes
    plt.subplot(2, 3, 3)
    plt.imshow(imagen_real)

    #Mostramos el histograma donde estan contabilizados todos los pixeles en escala de grises
    plt.subplot(2, 3, 4)
    plt.hist(img_g.flatten(), 100, [0,255], color = "b")

    #Mostramos el Rango[0,255]
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)

    plt.imshow(imagen_real)

    plt.show()
    cv2.imshow('Temperatura', imagen_real)  

#Salir con ESC
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
archivo.close()    
#Destruir la ventana y salir
cv2.destroyAllWindows()
quit()



