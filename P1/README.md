## Follow Line

This is the <a href="https://unibotics.org/academy/exercise/follow_line/">Follow Line</a> exercise from <a href="https://unibotics.org/academy/main">Unibotics</a> about creating a program that follows the red line in an F1 circuit.

It is based on the PID controller where PDI stands for:
- Proportional: $$u = -K_pe$$
- Derivative: $$u = -K_d\cdot \frac{de}{dt}$$
- Integral: $$u = -K_i\cdot \int e(t)dt$$

The formula for the PID is:
$$u = -K_pe-K_i\int_0^tedt-K_dde/dt$$