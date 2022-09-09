#CODIGO PRINCIPAL (Part 1)
import random
import math

#Configuracion
N_X = 28*28
N_Y = 40
N_Z = 10 
LRATE = 0.1

def crear_datos():
  #Descargar archivo  
  #Open, readlines, close
  r = open("../data/original/oneline.txt")
  ls = r.readlines()
  r.close()

  #Leer numero de numeros y medidas
  n_nums = int(ls.pop(0).strip())
  n_fils = int(ls.pop(0).strip())
  n_cols = int(ls.pop(0).strip())

  DAT = []
  RES = []
  #Procesar numero a numero
  for n in range(0,n_nums):
    #Numero correcto en string
    sres = ls.pop(0).strip()
    #Dibujo del numero en string
    sdat = ls.pop(0).strip()

    res = convertir_res(sres)
    RES.append(res)
    dat = convertir_dat(sdat)
    DAT.append(dat)

  return DAT,RES


DAT,RES = crear_datos()

def convertir_res(sres):
  res = []
  for z in range(0,N_Z):
    res.append(0.0)
  res[int(sres)] = 1.0
  return res

def convertir_dat(sdat):
  dat = []
  for c in sdat:
    dat.append(int(c))
  return dat

def crear_red():
  x = []          # primera capa
  for _ in range(0,N_X):
    x.append(0)

  w1 = []       #segunda capa
  for _ in range(0,N_Y):
    W = []
    for _ in range(0,N_X):
      r = (random.randint(0,100)/100)-0.5
      W.append(r)
    w1.append(W)


  #b1 #primer bias
  b1 = []
  for _ in range(0, N_Y):
    b1.append((random.randint(0,100)/100)-0.5)



  w2 = []       # pesos segunda capa
  for _ in range(0,N_Z):
    W = []
    for _ in range(0,N_Y):
      r = (random.randint(0,100)/100)-0.5
      W.append(r)
    w2.append(W)

  #b2      #segundo bias 
  b2 = []
  for _ in range(0, N_Z):
    b2.append((random.randint(0,100)/100)-0.5)




  return x, w1, w2,b1, b2

def get_predictions(A2):
    maxpos = -1
    maxval = -1.0
    for z in range(0,N_Z):
        if A2[z] > maxval:
            maxval = A2[z]
            maxpos = z
    return maxpos     #Devolvemos la posicion del maximo





def forward(x, w1, b1, w2, b2):    #FORWARD PROPAGATION
  Z1, A1 = forwardX2Y(x,w1,b1)
  Z2, A2 = forwardY2Z(w2,A1,b2)
  return Z1, A1, Z2, A2

def forwardX2Y(x,w1,b1):
  Z1 = []
  A1 = []
  for y in range(0,N_Y):
    z1, a1 =  forwardX2Yone(x,y,w1,b1)
    Z1.append(z1)
    A1.append(a1)
  return Z1, A1   #A1 es el output de la segunda capa

def forwardX2Yone(x,y,w1,b1):
  activation = 0
  for i in range(0,N_X):
    activation += x[i]*w1[y][i]

  activation += b1[y]
  return activation, max(activation, 0)     #Aplicamos funcion de activacion


def forwardY2Z(w2,A1,b2):
  Z2 = []
  A2 = []
  for z in range(0,N_Z):
    z2 = forwardY2Zone(z,w2,A1,b2)
    Z2.append(z2)
  A2 = softmax(Z2, A2)        #Aplicamos funcion Softmax
  return Z2, A2 #A2 es el output de la tercera capa
  
def forwardY2Zone(z,w2,A1, b2):
  activation = 0
  
  for i in range(0,N_Y):
    activation += A1[i]*w2[z][i]
  activation += b2[z]   

  return activation


def softmax(Z2,A2):
  A2 = []
  general = 0
  for z in range(0, N_Z):
    general += math.exp(Z2[z])
  for z in range(0, N_Z):
    A2.append(math.exp(Z2[z])/general)
  return A2



def mult_matrix(a,b):
    matrix = []
    for i in range(len(a)):
        matrix.append([])
        for j in range(len(b[0])):
            matrix[i].append(0)

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                matrix[i][j]  += a[i][k] * b[k][j]
    return matrix

def transpose(matrix):

    result = []
    for i in range(len(matrix[0])):
      partial = []
      for j in range(len(matrix)):
        partial.append(0)
      result.append(partial)

    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            result[i][j] = matrix[j][i] 
    return result


def back(Z1, A1, A2,  w2, x, y):   #back propagation ERROR
  dZ2, dW2 = backZ(A1, A2, y)
  dW1, dZ1 = backY(x, w2, Z1, dZ2)
  db2, db1 = backBias(dZ2,dZ1)
  return dW1, db1, dW2, db2
  

def backZ( A1,  A2,  y):
  dZ2 = []
  for z in range(0,N_Z):    
    #Error
    Error = [A2[z] - y[z]]
    dZ2.append(Error)

  dW2 = mult_matrix(dZ2, [A1])
  return dZ2, dW2

def backY(x, w2, Z1, dZ2):   #CALCULAMOS DZ1 Y DW1
  w2T = transpose(w2)
  dZ1 = mult_matrix(w2T, dZ2)
  for y in range(0,N_Y):  
    if Z1[y]>0:
      dZ1[y] = dZ1[y]
    else:
      dZ1[y] = [0]
  xT = [x]
  dW1 = mult_matrix(dZ1, xT)
  return dW1, dZ1
  
def backBias(dZ2, dZ1):
  db2 = 0
  for value in dZ2:
     db2 +=value[0]
  db1 = 0
  for value in dZ1:
     db1 +=value[0]

  return db2, db1

def propagation(w1,b1, w2,b2, dW1, dW2, db1, db2, LRATE, END):
  w1 = propagationY(w1, dW1, LRATE, END)
  b1 = propagationB1(b1,db1, LRATE, END)
  w2 = propagationZ(w2, dW2, LRATE, END)
  b2 = propagationB2(b2,db2, LRATE, END)
  return w1, b1,w2,b2

def propagationB1(b1, db1, LRATE, END):
  for i in range(len(b1)):
    b1[i] = b1[i] - LRATE*db1/END #dividimos por END para que sea el ponderado de todos los casos en una pasada
  return b1

def propagationB2(b2, db2, LRATE, END):
  for i in range(len(b2)):
    b2[i] = b2[i] - LRATE*db2/END #dividimos por END para que sea el ponderado de todos los casos en una pasada
  return b2

def propagationY(w1, dW1, LRATE, END):
  for y in range(0,N_Y):
    for x in range(0,N_X):
      w1[y][x] -= LRATE *dW1[y][x]/END #dividimos por END para que sea el ponderado de todos los casos en una pasada
  return w1

def propagationZ(w2, dW2, LRATE, END):
  for z in range(0,N_Z):
    for y in range(0,N_Y):
      w2[z][y] -= LRATE *dW2[z][y]/END #dividimos por END para que sea el ponderado de todos los casos en una pasada
  return w2

def predecir(X, w1,b1, w2, b2):
  _, _, _, A2 = forward(X,w1, b1, w2,  b2)
  #Buscar que Z es el numero maximo
  maxpos = get_predictions(A2)
  return maxpos




#Crear red y datos  !wget -O oneline.txt https://www.dropbox.com/s/ps958u765f2jabe/oneline.txt?dl=0

def entrenar(nums,x,  w1, w2,b1, b2):
  
  #Para cada numero de mis datos
  END = 20000
  for i in range(nums):
    corrects = 0
    for n in range(0,END):
      x = DAT[n]
      y = RES[n]

      Z1, A1, Z2, A2  = forward(x, w1, b1, w2, b2)      # generamos el output
      dW1, db1, dW2, db2 = back(Z1, A1, A2,  w2, x, y)  # back propagation
      w1, b1, w2, b2 = propagation(w1,b1, w2,b2, dW1, dW2,db1, db2, LRATE,END)   #update weights
      if i%10 == 0:
        if A2.index(max(A2)) == y.index(max(y)):
          corrects +=1

    if i %10 == 0:
      print("Round",i)
      print("average hasta ahora de:" ,corrects/END)
      corrects = 0
  return w1, b1, w2, b2

#CODIGO PRINCIPAL (Part 2)

#Entrenarx
X,  w1, w2, b1, b2 = crear_red()        
w1, b1,w2,b2 = entrenar(500, X,  w1, w2, b1, b2) 


#Predecir
X = convertir_dat(
"0000000000000000000000000000"+
"0000000000000000000000000000"+
"0000000000000000000000000000"+
"0000000000000000000000000000"+
"0000000000000000000000000000"+
"0000000000000000000000000000"+
"0000000000000001111100000000"+
"0000000000000011111110000000"+
"0000000000001111101110000000"+
"0000000000001110001100000000"+
"0000000000011110011100000000"+
"0000000000111100011100000000"+
"0000000000111000111100000000"+
"0000000000111001111000000000"+
"0000000000110011111000000000"+
"0000000001110111110000000000"+
"0000000001111111100000000000"+
"0000000000111111100000000000"+
"0000000000001111000000000000"+
"0000000000001110000000000000"+
"0000000000001110000000000000"+
"0000000000011100000000000000"+
"0000000000111100000000000000"+
"0000000000111000000000000000"+
"0000000000111000000000000000"+
"0000000000110000000000000000"+
"0000000000000000000000000000"+
"0000000000000000000000000000")
num = predecir(X,w1,b1,w2,b2)
print ("El numero es un", num, "y era un ", 9)



def Read_Weights():
    w1 = []
    with open('../data/weights/w1.txt','r') as file:
        for line in file:
            # print(line)
            lista = line.strip().split(',')
            intermediate = []
            for value in lista:
                intermediate.append(float(value))
            w1.append(intermediate)
    w2 = []
    with open('../data/weights/w2.txt','r') as file:
        for line in file:
            # print(line)
            lista = line.strip().split(',')
            intermediate = []
            for value in lista:
                intermediate.append(float(value))
            w2.append(intermediate)


    b1 = []
    with open('../data/weights/b1.txt','r') as file:
        for line in file:
            # print(line)
            lista = line.strip().split(',')
            for value in lista:
                b1.append(float(value))

    b2 = []
    with open('../data/weights/b2.txt','r') as file:
        for line in file:
            # print(line)
            lista = line.strip().split(',')
            for value in lista:
                b2.append(float(value))
    return w1, w2, b1, b2 

w1, w2, b1, b2 = Read_Weights()

X_test = DAT[42000:]
y_test = RES[42000:]
number = 0
for i in range(len(y_test)):  
  X = X_test[i]
  E = y_test[i]
  _, _,_, A2 = forward(X,w1, b1, w2, b2)
  if get_predictions(A2) == get_predictions(E):
    number +=1
print(f"{round(number/len(y_test),3)*100}%")
w1_save = ''
for i in range(len(w1)):
    w1_save += f"{w1[i][0]}" 
    for x in range(1,len(w1[i])):
        w1_save += f",{w1[i][x]}"
    w1_save +='\n'
with open('../w1.txt','w') as file:
    file.write(w1_save)

w2_save = ''
for i in range(len(w2)):
    w2_save += f"{w2[i][0]}" 
    for x in range(1,len(w2[i])):
        w2_save += f",{w2[i][x]}"
    w2_save +='\n'
with open('../w2.txt','w') as file:
    file.write(w2_save)
    
b1_save = ''
b1_save += f"{b1[0]}" 
for x in range(1,len(b1)):
    b1_save += f",{b1[x]}"
with open('../b1.txt','w') as file:
    file.write(b1_save)

b2_save = ''
b2_save += f"{b2[0]}" 
for x in range(1,len(b2)):
    b2_save += f",{b2[x]}"
with open('../b2.txt','w') as file:
    file.write(b2_save)