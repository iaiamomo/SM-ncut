import cv2
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from scipy.sparse import linalg

#sigmaI=4, sigmaX=6, l=0
def create_graph(img):
  h, w = img.shape[:2]

  g = nx.Graph()

  n = 0
  sigmaI = 4
  sigmaX = 6

  for i in range(h):
    for j in range(w):
      distance = 1
      if i == h-1 and j != w-1:
        color1 = abs(int(img[i,j]) - int(img[i,j+1]))
        weight1 = math.exp(-(color1**2)/(sigmaI**2)) * math.exp(-(distance**2)/(sigmaX**2))
        g.add_edge(n, n+1,  weight = weight1)
      elif j == w-1 and i != h-1:
        color2 = abs(int(img[i,j]) - int(img[i+1,j]))
        weight2 = math.exp(-(color2**2)/(sigmaI**2)) * math.exp(-(distance**2)/(sigmaX**2))
        g.add_edge(n, n+w, weight = weight2)
      elif j != w-1 and i != h-1:
        color1 = abs(int(img[i,j]) - int(img[i,j+1]))
        weight1 = math.exp(-(color1**2)/(sigmaI**2)) * math.exp(-(distance**2)/(sigmaX**2))
        g.add_edge(n, n+1,  weight = weight1)
        color2 = abs(int(img[i,j]) - int(img[i+1,j]))
        weight2 = math.exp(-(color2**2)/(sigmaI**2)) * math.exp(-(distance**2)/(sigmaX**2))
        g.add_edge(n, n+w, weight = weight2)
      n+=1

  return g

# find 4 smallest EIGENVECTORS
def find_eigenvectors(g):

  # adjacency matrix, degree matrix and laplacian matrix
  adj = nx.adjacency_matrix(g)
  degr = np.diag(np.sum(np.array(adj.todense()), axis=1))
  lapl = degr - adj

  # compute eigenvalues and eigenvectors
  e,v = np.linalg.eigh(lapl)

  # indexes of sorted eigenvalues 
  idx = np.argsort(e)

  # find 4 smallest eigenvalues
  idsecond = 0
  idthird = 0
  idfourth = 0
  idfifth = 0
  for i in idx:
    if e[idx[i]] > 0.0:
      idsecond = i
      break
  vect2 = v[:,idx[idsecond]]
  for i in idx:
    if e[idx[i]] > e[idx[idsecond]]:
      idthird = i
      break
  vect3 = v[:,idx[idthird]]
  for i in idx:
    if e[idx[i]] > e[idx[idthird]]:
      idfourth = i
      break
  vect4 = v[:,idx[idfourth]]
  for i in idx:
    if e[idx[i]] > e[idx[idfourth]]:
      idfifth = i
      break
  vect5 = v[:,idx[idfifth]]

  vect = [vect2, vect3, vect4, vect5]

  return vect

# create the image depending on the cut
def create_image(svect, new_image, color, img):
  h, w = img.shape[:2]

  # splitting point
  l = 0

  s = []
  cntrl = 0
  for elem in svect:
    if elem > l:
      s.append(cntrl)
      x = cntrl/w
      prod = w*x
      y = cntrl-prod
      new_image[x,y] = tuple(color)
    cntrl+=1

  return new_image, s

def main(name):
  # load the image
  img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
  
  # create new image to visualize the cuts
  imgseg1 = cv2.imread(name)
  imgseg1 = cv2.cvtColor(imgseg1, cv2.COLOR_BGR2RGB)

  imgseg2 = cv2.imread(name)
  imgseg2 = cv2.cvtColor(imgseg2, cv2.COLOR_BGR2RGB)
  
  imgseg3 = cv2.imread(name)
  imgseg3 = cv2.cvtColor(imgseg3, cv2.COLOR_BGR2RGB)
  
  imgseg4 = cv2.imread(name)
  imgseg4 = cv2.cvtColor(imgseg4, cv2.COLOR_BGR2RGB)

  # create the graph and compute the segmentation
  g = create_graph(img)
  vec = find_eigenvectors(g)  #4 smallest eigenvectors
  imgseg1, s1 = create_image(vec[0], imgseg1, (255,0,0), img)
  imgseg2, s2 = create_image(vec[1], imgseg2, (0,255,0), img)
  imgseg3, s3 = create_image(vec[2], imgseg3, (0,0,255), img)
  imgseg4, s4 = create_image(vec[3], imgseg4, (0,255,255), img)

  plt.figure(1)
  plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
  plt.figure(2)
  plt.imshow(imgseg1)
  plt.figure(3)
  plt.imshow(imgseg2)
  plt.figure(4)
  plt.imshow(imgseg3)
  plt.figure(5)
  plt.imshow(imgseg4)
  plt.show()

#Diodato 100x57
main('DiodatoResized.jpg')