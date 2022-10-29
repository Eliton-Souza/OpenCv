#Aluno: Eliton Souza
#Matricula: 

import cv2
from numpy import *
import math
from matplotlib import pyplot as plt
import random
import numpy as np


img_original= "Imagens\Originais\Lena.png"
local_alterada= "Imagens\Alteradas"


img = cv2.imread(img_original)
img2 = cv2.imread(img_original)
img3 = cv2.imread(img_original)
img4 = cv2.imread(img_original)

vermelho = zeros((256), dtype=int)
verde = zeros((256), dtype=int)
azul = zeros((256), dtype=int)
logarimo = zeros(256,dtype = int)

linha = len(img)
coluna = len(img[1])
nove = zeros((3,3),dtype=int)
nove2 = zeros((3,3),dtype=int)
vet = zeros(9,dtype=int)
total = img.shape[0]*img.shape[1]
cinza1 = zeros((256), dtype=int)
cinza2 = zeros((256), dtype=int)
cinza3 = zeros((256), dtype=int)
cinza4 = zeros((256), dtype=int)
interior = zeros(256,dtype = int)
borda = zeros(256,dtype = int)

def imprime_opcoes():

    print("\nOpçoes:\n1-Alterar brilho\n2-Imagem negativa\n3-Histograma global\n4-Histograma local\n5-Transformadas radiométricas\n6-Tres filtros espaciais\n7-Detecção de bordas\n8-BIC\n9-Sair\n")
    p = int(input())
    return p

p= imprime_opcoes()

while((p!=9)):
    
    if(p==1):

        tam_brilho = int(input("Digite a portentagem de brilho da imagem: "))
        aux= tam_brilho
        
        cv2.imshow('Imagem original', img)
        cv2.waitKey(0)

        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                tam_brilho= aux

                (b,g,r) = img[i,j]

                if(tam_brilho > 50 and tam_brilho <= 100):
                    tam_brilho -= 50

                    b += (255 - b) * tam_brilho /50                   
                    g += (255 - g) * tam_brilho /50
                    r += (255 - r) * tam_brilho /50
                   
                elif(tam_brilho < 50 and tam_brilho >= 0):
                    b = tam_brilho * b/50
                    g = tam_brilho * g/50
                    r = tam_brilho * r/50

                img[i,j] = (b, g, r)
              

        cv2.imwrite(local_alterada+"\\Brilho.png", img)
        print("Imagem salva")

        cv2.imshow('Imagem com brilho alterado', img)
        cv2.waitKey(0)

        cv2.destroyAllWindows() 


    elif(p==2):

        cv2.imshow('Imagem original', img)
        cv2.waitKey(0)

        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                (g, b, r) = img[i, j]
                g = 255-g
                b = 255-b
                r = 255-r
                img[i,j] = (g, b, r)


        cv2.imwrite(local_alterada+"\\Negativa.png", img)
        print("Imagem salva")

        cv2.imshow('Imagem negativa', img)
        cv2.waitKey(0)

        cv2.destroyAllWindows() 


    elif(p==3):

        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                (b, g, r) = img[i, j]

                vermelho[r] = vermelho[r] + 1
                verde[g] = verde[g] + 1
                azul[b] = azul[b] + 1

        arquivo = open("Histograma Global.txt", 'w')

        arquivo.write("Vetor representado cor vermelha" + "\n")
        arquivo.write(str(vermelho) + '\n \n')
        arquivo.write("Vetor representado cor azul" + "\n")
        arquivo.write(str(verde)+ '\n \n')
        arquivo.write("Vetor representado cor verde" + "\n")
        arquivo.write(str(azul)+ '\n')
        arquivo.close()
        print("Arquivo criado \n")

        canais = cv2.split(img)
        cores = ("b", "g", "r")

        plt.figure()
        plt.title("Histograma Colorido")
        plt.xlabel("Intensidade")
        plt.ylabel("Número de Pixels")

        for (canal, cor) in zip(canais, cores):
            hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
            plt.plot(hist, cor)
            plt.xlim([0, 256])

        plt.show()


    elif(p==4):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        for i in range(0, (len(img)//2)-1):
            for j in range(0, (len(img[1])//2)-1):
                b = img[i,j]
                cinza1[b] = cinza1[b] + 1
            

        for i in range(0, (len(img)//2) - 1):
            for j in range(len(img[1])//2, len(img[1])):
         

                b = img[i,j]
                cinza2[b] = cinza2[b] + 1
            


        for i in range(len(img)//2, len(img)):
            for j in range(0, (len(img[1])//2)-1):
          

                b = img[i,j]
                cinza3[b] = cinza3[b] + 1
            


        for i in range(len(img)//2, len(img)):
            for j in range(len(img[1])//2, len(img[1])):
           
                
                b = img[i,j]
                cinza4[b] = cinza4[b] + 1
            

        D = [cinza1 + cinza2 + cinza3 + cinza4 for cinza1, cinza2, cinza3, cinza4 in zip(cinza1, cinza2, cinza3,cinza4)]

        arquivo = open("Histograma Local - Partição.txt", 'w')

        arquivo.write("Vetor representando a Partição 1:" + "\n")
        arquivo.write(str(cinza1) + '\n \n')
        arquivo.write("Vetor representando a Partição 2:" + "\n")
        arquivo.write(str(cinza2)+ '\n \n')
        arquivo.write("Vetor representando a Partição 3:" + "\n")
        arquivo.write(str(cinza3)+ '\n \n')
        arquivo.write("Vetor representando a Partição 4:" + "\n")
        arquivo.write(str(cinza4)+ '\n \n')
        arquivo.write("Partições em um unico vetor:" + "\n")
        arquivo.write(str(D)+ '\n')
        arquivo.close()
        print("Arquivo criado \n")


    elif(p==5):
        

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        total = img.shape[0]*img.shape[1]
        a = 0
        cont =0

        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                b = img[i,j]
                if(a<b):
                    a = b
                
        cinza = zeros((a+1), dtype=float)
        for i in range(0, len(img)):
                for j in range(0, len(img[1])):
                    c = img[i,j]
                   
        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                cinza[img[i,j]]= cinza[img[i,j]]+1

        for i in range(0,a+1):
            cinza[i] = (float)((cinza[i])/(total))

        for i in range(1,(a+1)):
            cinza[i] = cinza[i] + cinza[i-1]

        for i in range(0,(a+1)):
             cinza[i] = a*cinza[i]
             
        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                img[i,j] = cinza[img[i,j]]



        cv2.imwrite(local_alterada+"\\TR Equalizado.png", img)

        for i in range(0, linha):
            for j in range(0, coluna):

                a = 255/math.log(255)
                c = img2[i,j]
                y = a*math.log(c+1)
                img2[i,j] = y
                c = img2[i,j]
                logarimo[c] = logarimo[c] + 1

        cv2.imwrite(local_alterada+"\\TR Logaritmo.png", img2)

        for i in range(0, len(img)):
            for j in range(0, len(img[1])):
                a = 255/math.sqrt(255)
                c = img3[i,j]
                y = a*math.sqrt(c)
                
                img3[i,j] = y
            
                c = img3[i,j]
                logarimo[c] = logarimo[c] + 1

        cv2.imwrite(local_alterada+"\\TR Quadrada.png", img3)

        cv2.imshow('Equaliza', img)
        cv2.imshow('Logaritmo', img2)
        cv2.imshow('Quadrada', img3)

        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        
    elif(p==6):


        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(img3)-1):
            for j in range(1, len(img3[1])-1):
                
                nove[0][0] = img3[i-1][j-1]
                nove[0][1] = img3[i-1][j]
                nove[0][2] = img3[i-1][j+1]
                nove[1][0] = img3[i][j-1]
                nove[1][1] = img3[i][j]
                nove[1][2] = img3[i][j+1]
                nove[2][0] = img3[i+1][j-1]
                nove[2][1] = img3[i+1][j]
                nove[2][2] = img3[i+1][j+1]
                k = nove[0][0] + nove[0][1] +  nove[0][2] + nove[1][0] + nove[1][1] + nove[1][2] +  nove[2][0] + nove[2][1] + nove[2][2]
                k = k/9

                if(abs(k - img3[i][j])>10):
                    img3[i][j] = k
                else:
                     img3[i][j] =img3[i][j]
                     
        cv2.imwrite(local_alterada+"\\Filtro Media.png", img3)

        k = int(0.10 * total)

        for i in range(0,k):
            x = random.randint(0,linha-1)
            l = random.randint(0,coluna-1)
            img[x,l] = 0

            
        for i in range(1, len(img)-1):
            for j in range(1, len(img[1])-1):
                
                nove[0][0] = img[i-1][j-1]
                nove[0][1] = img[i-1][j]
                nove[0][2] = img[i-1][j+1]
                nove[1][0] = img[i][j-1]
                nove[1][1] = img[i][j]
                nove[1][2] = img[i][j+1]
                nove[2][0] = img[i+1][j-1]
                nove[2][1] = img[i+1][j]
                nove[2][2] = img[i+1][j+1]
                k = nove[0][0] + nove[0][1] +  nove[0][2] + nove[1][0] + nove[1][1] + nove[1][2] +  nove[2][0] + nove[2][1] + nove[2][2]
                k = k/9

                if(abs(k - img[i][j])>10):
                    img[i][j] = k
                else:
                     img[i][j] =img[i][j]


       

        cv2.imwrite(local_alterada+"\\Filtro Pimenta.png", img)

        for i in range(1, len(img4)-1):
            for j in range(1, len(img4[1])-1):
                
                vet[0] = img4[i-1][j-1]
                vet[1] = img4[i-1][j]
                vet[2] = img4[i-1][j+1]
                vet[3] = img4[i][j-1]
                vet[4] = img4[i][j]
                vet[5] = img4[i][j+1]
                vet[6] = img4[i+1][j-1]
                vet[7] = img4[i+1][j]
                vet[8] = img4[i+1][j+1]
                vet.sort()
                k = (vet[2]+vet[3]+vet[4]+vet[5]+vet[6]+vet[7]+vet[8])
                k = k/7
                
                img4[i][j] = k

        cv2.imwrite(local_alterada+"\\Filtro K Vizinhos.png", img4)

        k = int(0.10 * total)
        for i in range(0,k):
            x = random.randint(0,linha-1)
            l = random.randint(0,coluna-1)
            img2[x,l] = 255



        for i in range(1, len(img2)-1):
            for j in range(1, len(img2[1])-1):
                
                vet[0] = img2[i-1][j-1]
                vet[1] = img2[i-1][j]
                vet[2] = img2[i-1][j+1]
                vet[3] = img2[i][j-1]
                vet[4] = img2[i][j]
                vet[5] = img2[i][j+1]
                vet[6] = img2[i+1][j-1]
                vet[7] = img2[i+1][j]
                vet[8] = img2[i+1][j+1]
                vet.sort()
                k = (vet[2]+vet[3]+vet[4]+vet[5]+vet[6]+vet[7]+vet[8])
                k = k/7
                
                img2[i][j] = k
        
        
        cv2.imwrite(local_alterada+"\\Filtro Sal.png", img2)


        cv2.imshow('Filtro Media', img3)
        cv2.imshow('Filtro Pimenta', img)
        cv2.imshow('Filtro K Vizinhos', img4)
        cv2.imshow('Filtro Sal', img2)
        cv2.waitKey(0)

        cv2.destroyAllWindows() 
     

    elif(p==7):
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = 30
        imgQuant = np.uint8(img / r) * r
        for i in range(0, len(imgQuant)-1):
            for j in range(0, len(imgQuant[1])-1):
                nove2[0][0] = imgQuant[i][j]
                nove2[0][1] = imgQuant[i][j+1]
                nove2[1][0] = imgQuant[i+1][j]
                nove2[1][1] = imgQuant[i+1][j+1]
                k = abs(nove2[0,0]-nove2[1][1])+abs(nove2[0][1]-nove2[1][0])

                if(k>50):
                    imgQuant[i][j] = 0
                else:
                     imgQuant[i][j] = 255

        cv2.imwrite(local_alterada+"\\Bordas.png", imgQuant)

        cv2.imshow('Bordas', imgQuant)
        cv2.waitKey(0)

        cv2.destroyAllWindows() 


    elif(p==8):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = 64
        imgQuant = np.uint8(img / r) * r
        
        for i in range(1, len(imgQuant)-1):
            for j in range(1, len(imgQuant[1])-1):

                c = imgQuant[i,j]
                if((c == imgQuant[i-1][j] )and(c == imgQuant[i][j+1])and(c == imgQuant[i+1][j])and(c == imgQuant[i][j-1])):

                    interior[c] += 1
            
                else:
            
                    borda[c] += 1

        cv2.imshow("imagem",imgQuant)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        arquivo = open("Bic.txt", 'w')

        arquivo.write("Vetor representando a Partição 1:" + "\n")
        arquivo.write(str(interior) + '\n \n')
        arquivo.write("Vetor representando a Partição 2:" + "\n")
        arquivo.write(str(borda)+ '\n \n')

        arquivo.close()



    p= imprime_opcoes()
    img = cv2.imread(img_original)
    img2 = cv2.imread(img_original)
    img3 = cv2.imread(img_original)
    img4 = cv2.imread(img_original)
    
print("Encerrado")
