from tkinter import E
import matplotlib.pyplot as plt

file = open("data.dat", "r")
content = file.read()
file.close()

content = [ (x.split()) for x in content.split('\n') if x != [] ]
x = [ float( x[0] ) for x in content]
y = [ float( y[1] ) for y in content ]

plt.scatter(x, y)

file1 = open("line.dat", "r")
content = file1.read()
file1.close()

w1 = float( content.split()[0] )
w2 = float( content.split()[1] )
bias = float( content.split()[2] )

a = - w1 / w2
b = - bias / w2
print( f"y = {a}x + {b}" )

fun = lambda x: a*x + b

x = range(-5, 15)
y = [ fun(i) for i in x ]

plt.plot(x ,y, color='red')
plt.ylim(-10, 20)
plt.show()
