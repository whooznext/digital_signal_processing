import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

N = 20
M = 5

# (a) plotting discrete signals

n = np.arange(0, N-1)


un = (n >= 0).astype(float)

xn = (1 / 2) ** n * un


hn = (n <= M).astype(float) * (1 / (M + 1))

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

yn = sig.convolve(xn, hn)
nn = np.arange(0, len(yn))


plt.figure(1)
plt.title('convolution of x[n] and h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.stem(nn, yn)

# (c) computing DTFT
pi=np.pi

W=np.linspace(-3*pi,3*pi,100)


temp = np.zeros((len(n), len(W))).astype(complex)


for i in n:
    temp[i:,] = (np.array([np.exp(-1j*W)])**i)

'''for i in nn:
    temp_conv[i:,] = (np.array([np.exp(-1j*W)])**i)'''
X = xn.dot(temp)
H = hn.dot(temp)
Y = X*H


plt.figure(2)
plt.subplot(2,1,1)
plt.title('DTFT of x[n]')
plt.ylabel('Magnitude $|X(e^{j\omega})|$')
plt.plot(W, abs(X))


plt.subplot(2,1,2)
plt.xlabel('$\omega$')
plt.ylabel('Phase $<X(e^{j\omega})$')
plt.plot(W, np.angle(X))

plt.figure(3)
plt.subplot(3,1,1)
plt.title('DTFT magnitudes')
plt.ylabel('$|X(e^{j\omega})|$')
plt.plot(W, abs(X))

plt.subplot(3,1,2)
plt.ylabel('$|H(e^{j\omega})|$')
plt.plot(W, abs(H))

plt.subplot(3,1,3)
plt.ylabel('$|Y(e^{j\omega})|$')
plt.plot(W, abs(Y))
plt.xlabel('$\omega$')

plt.subplots_adjust(hspace=0.4)

# display the plot
plt.show()


