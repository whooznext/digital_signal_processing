import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

N=20
M=5
# (a) plotting discrete signals
n=np.arange(0,N-1)


un = (n>=0).astype(float)
xn = un * (0.5)**n
hn = un * (1.0/(M+1))*(n<=M).astype(float)

plt.figure(0)

plt.subplot(2,1,1)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.stem(n, xn)

plt.subplot(2,1,2)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.stem(n, hn)

# (b) discrete convolution
yn=sig.convolve(xn, hn)
nn=np.arange(0,len(yn))


plt.figure(1)
plt.title('convolution of x[n] and h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.stem(nn,yn)

# (c) computing DTFT

pi=np.pi

W=np.linspace(-3*pi,3*pi,100)


x_matrix = []
h_matrix = []
y_matrix = []
for row in W:
    for col in n:
        x_matrix.append(np.exp(-1*1j*row*col))
        h_matrix.append(np.exp(-1*1j*row*col))

for row in W:
    for col in nn:
        y_matrix.append(np.exp(-1*1j*row*col))

x_matrix = np.reshape(x_matrix, (len(W), len(n)))
h_matrix = np.reshape(h_matrix, (len(W), len(n)))
y_matrix = np.reshape(y_matrix, (len(W), len(nn)))

X = np.matmul(x_matrix, xn)
H = np.matmul(h_matrix, hn)
Y = np.matmul(y_matrix, yn)

plt.figure(2)
plt.subplot(2,1,1)
plt.title('DTFT of x[n]')
plt.ylabel('Magnitude $|X(e^{j\omega})|$')
plt.plot(W,abs(X))

plt.subplot(2,1,2)
plt.xlabel('$\omega$')
plt.ylabel('Phase $<X(e^{j\omega})$')
plt.plot(W,np.angle(X))

plt.figure(3)
plt.subplot(3,1,1)
plt.title('DTFT magnitudes')
plt.ylabel('$|X(e^{j\omega})|$')
plt.plot(W,abs(X))

plt.subplot(3,1,2)
plt.ylabel('$|H(e^{j\omega})|$')
plt.plot(W,abs(H))

plt.subplot(3,1,3)
plt.ylabel('$|Y(e^{j\omega})|$')
plt.plot(W,abs(Y))
plt.xlabel('$\omega$')

plt.subplots_adjust(hspace=0.4)


plt.show()



