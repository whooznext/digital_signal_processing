import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

# define LCCDE
a=[1,-5/6,5/12,-1/12]
b=[1,0.5]
dt=1

plt.figure(0)


n,h=sig.dimpulse((b,a,dt),n=14)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.stem(n,np.squeeze(h))
plt.grid()

plt.figure(1)


w,H=sig.freqz(b,a)
plt.subplot(2,1,1)
plt.plot(w,abs(H))
plt.xlabel('$\omega$')
plt.ylabel('Magnitude')
plt.grid()


plt.subplot(2,1,2)
angles = np.unwrap(np.angle(H))
plt.plot(w, angles)
plt.xlabel('$\omega$')
plt.ylabel('Phase (radians)')
plt.grid()
plt.axis('tight')


z,p,k=sig.tf2zpk(b,a)

plt.subplots_adjust(hspace=0.4)

plt.figure(2)
ax=plt.gca()
plt.scatter(p.real,p.imag,marker='x',edgecolors='b')
plt.scatter(z.real,z.imag,marker='o',edgecolors='r',facecolors='none')

circ=plt.Circle((0,0),1,fill=False,color='black', ls='solid', alpha=0.1)
ax.add_patch(circ)
r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
plt.axis('scaled')
plt.axis([-r, r, -r, r])
plt.grid()
plt.title('pole-zero plot')

plt.show()
