set xrange [0:1]
set yrange [:]
set terminal "png"
set output "3rd_poly.png"
plot f(x)=a*x**3+(1-a)*x, \
     a=1, f(x), \
     a=0.9, f(x), \
     a=0.8, f(x), \
     a=0.7, f(x), \
     a=0.6, f(x), \
     a=0.5, f(x), \
     a=0.4, f(x), \
     a=0.3, f(x), \
     a=0.2, f(x), \
     a=0.1, f(x), \
     a=0, f(x), \
     a=-0.1, f(x), \
     a=-0.2, f(x), \
     a=-0.3, f(x), \
     a=-0.4, f(x), \
     a=-0.5, f(x)
set yrange [:]
set output "3rd_poly_deriv.png"
plot f(x)=2*a*x**2+(1-a), \
     a=1, f(x), \
     a=0.9, f(x), \
     a=0.8, f(x), \
     a=0.7, f(x), \
     a=0.6, f(x), \
     a=0.5, f(x), \
     a=0.4, f(x), \
     a=0.3, f(x), \
     a=0.2, f(x), \
     a=0.1, f(x), \
     a=0, f(x), \
     a=-0.1, f(x), \
     a=-0.2, f(x), \
     a=-0.3, f(x), \
     a=-0.4, f(x), \
     a=-0.5, f(x)

set xrange [-90:90]
set output "x_from_angle.png"
plot sin(pi/180*x), cos(pi/180*x), sin(pi/180*x)* sin(pi/180*x)+ cos(pi/180*x)* cos(pi/180*x)

set xrange [-90:90]
set output "polar_3rd_poly.png"
plot f(x)=a*sin(pi/180*x)**3+(1-a)*sin(pi/180*x), \
     a=1, f(x), \
     a=0.8, f(x), \
     a=0.6, f(x), \
     a=0.4, f(x), \
     a=0.2, f(x), \
     a=0, f(x), \
     a=-0.2, f(x), \
     a=-0.4, f(x), \
     a=-0.6, f(x)
