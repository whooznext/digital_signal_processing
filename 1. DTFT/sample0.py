import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

N=20
M=5

# (a) plotting discrete signals
n=np.arange(0,N-1)

n=np.array([n]).T

a = 0.5
exp_signal = (a)**n
un = (n>=0).astype(float)

xn = exp_signal * un
hn = (n<=M).astype(float) * (1/(M+1))


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

yn = sig.convolve(xn,hn)
nn = np.arange(0,len(yn))


nn=np.array([nn]).T

plt.figure(1)
plt.title('convolution of x[n] and h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.stem(nn,yn)

# (c) computing DTFT
pi=np.pi

W=np.linspace(-3*pi,3*pi,100)


W=np.array([W]).T
ejw= np.exp(-1j*W)

X = 1/(1-(0.5*ejw))
H = (1/(M+1)) * (np.sin(W*(M+1)/2)) * (np.exp(-1j*(M*W/2))) * (1/np.sin(W/2))
Y = np.multiply(X,H)



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


