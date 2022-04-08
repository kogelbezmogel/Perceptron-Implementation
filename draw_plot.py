import matplotlib.pyplot as plt

file = open("data.dat", "r")
content = file.read()
file.close()

content = [ (x.split()) for x in content.split('\n') ]
x0 = [ float( x[0] ) for x in content if float(x[2]) == 1.0 ]
y0 = [ float( y[1] ) for y in content if float(y[2]) == 1.0 ]
x1 = [ float( x[0] ) for x in content if float(x[2]) == -1.0 ]
y1 = [ float( y[1] ) for y in content if float(y[2]) == -1.0 ]

plt.scatter(x0, y0, color='blue')
plt.scatter(x1, y1, color='green')

file1 = open("line.dat", "r")
content = file1.read()
file1.close()

try :
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
except :
    print("Function can not be inerprated")


plt.ylim(-10, 20)
plt.xlim(-10, 20)
plt.show()
