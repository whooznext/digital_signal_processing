import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

W1 = 100

N = 20
M = 5

a = 0.5

# (a) plotting discrete signals

n = np.arange(0, N, 1)
n1 = np.arange(0, N, 1)

en = (a) ** n


un = (n >= 0).astype(float)
un2 = (n1 >= M + 1).astype(float)
un3 = (n1 >= 0).astype(float)
un1 = un3 - un2

xn = en * un
hn = un1 * (1 / (M + 1))


plt.figure(0)

plt.subplot(2, 1, 1)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.stem(n, xn)

plt.subplot(2, 1, 2)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.stem(n1, hn)

# (b) discrete convolution

yn = sig.convolve(xn, hn, mode='full')

nn = np.arange(0, len(yn))



plt.figure(1)
plt.title('convolution of x[n] and h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.stem(nn, yn)

# (c) computing DTFT
pi = np.pi

W = np.linspace(-3 * pi, 3 * pi, 100)


X = [0] * W1
for k in range(W1):
    s = 0
    p = 0
    for t in range(N):
        s += xn[t] * np.exp(-1j * W[k] * t)
    X[k] = s
X = np.array([X]).T


H = [0] * W1
for k in range(W1):
    s = 0
    p = 0
    for t in range(N):
        s += hn[t] * np.exp(-1j * W[k] * t)
    H[k] = s
H = np.array([H]).T


Y = [0] * W1
for k in range(W1):
    s = 0
    p = 0
    for t in range(N):
        s += yn[t] * np.exp(-1j * W[k] * t)
    Y[k] = s
Y = np.array([Y]).T

plt.figure(2)
plt.subplot(2, 1, 1)
plt.title('DTFT of x[n]')
plt.ylabel('Magnitude $|X(e^{j\omega})|$')
plt.plot(W, abs(X))

plt.subplot(2, 1, 2)
plt.xlabel('$\omega$')
plt.ylabel('Phase $<X(e^{j\omega})$')
plt.plot(W, np.angle(X))

plt.figure(3)
plt.subplot(3, 1, 1)
plt.title('DTFT magnitudes')
plt.ylabel('$|X(e^{j\omega})|$')
plt.plot(W, abs(X))

plt.subplot(3, 1, 2)
plt.ylabel('$|H(e^{j\omega})|$')
plt.plot(W, abs(H))

plt.subplot(3, 1, 3)
plt.ylabel('$|Y(e^{j\omega})|$')
plt.plot(W, abs(Y))
plt.xlabel('$\omega$')

plt.subplots_adjust(hspace=0.4)

# display the plot
plt.show()
