import numpy as np

def Householders(A):
  n= A.shape[0]
  P=np.eye(n)
  b=1
  for i in range(n):
    for j in range(n):
      vector=A[j:n,j][1:n]
      if np.sign(vector[0])==-1:
        b=1
      else:
        b=-1

      a=b*np.linalg.norm(vector)
      u=vector-(a*np.eye(vector.shape[0])[:,0])
      w=-1*u/np.linalg.norm(u)
      H=np.eye(n-j-1)-2*np.outer(w,w)
      T=np.eye(n)
      T[j+1:n, j+1:n]=H
      A=T@A@T.T
      P=P@T
      if vector.shape[0]==1:
        break
      

      
  return A,P


A = np.array([[7, 2, 3, -1],
              [2, 8, 5, 1],
              [3, 5, 12, 9],
              [-1, 1, 9, 7]], dtype=float)


print(Householders(A)[0],"\n","\n",Householders(A)[1])