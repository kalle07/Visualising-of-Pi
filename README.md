
# Visualising the Digits of π as a Two-Dimensional Trajectory

## 1. Purpose

This program transforms the decimal digits of π into a geometric trajectory.


example: 0111111111

<img width="1802" height="1239" alt="grafik" src="https://github.com/user-attachments/assets/fd271dbc-8327-48e4-99fc-a88d0a2b8b89" />


example: 019191919

<img width="1802" height="1239" alt="grafik" src="https://github.com/user-attachments/assets/8ecfa664-54cc-4ac8-b391-de8baea48ab8" />


now Pi: 200_000 digits, start in coordinate origin, a spectrum ranging from red to red to estimate the direction/trace

<img width="1802" height="1239" alt="grafik" src="https://github.com/user-attachments/assets/553a7e97-a6f6-4273-be75-51c4e19f36c9" />


now Pi: 1_000_000 digits, start in coordinate origin, a spectrum ranging 10 times from red to red to estimate the direction/trace (white box first 200k digits)

<img width="1720" height="1210" alt="grafik" src="https://github.com/user-attachments/assets/fede9043-5aba-47e4-9a69-390f60cd7ef2" />


Each digit is interpreted as an angular increment. These increments are accumulated to determine the current direction of movement. After each angular update, the program moves exactly one unit in that direction.

The result is a continuous two-dimensional path whose movements are determined entirely by the sequence of digits of π.

The purpose is to make the numerical sequence visually accessible and explore the geometric structures that emerge from it.

## 2. The basic principle: Ten possible angular increments

A complete circle contains 360 degrees. The ten decimal digits, from 0 to 9, divide this circle into ten equal angular increments:

\[
\frac{360^\circ}{10}=36^\circ
\]

Each digit represents the following change in direction:

| Digit | Angular increment |
|---:|---:|
| 0 | 0° |
| 1 | 36° |
| 2 | 72° |
| 3 | 108° |
| 4 | 144° |
| 5 | 180° |
| 6 | 216° |
| 7 | 252° |
| 8 | 288° |
| 9 | 324° |

**Important:** These values are angular changes, not absolute movement directions. The actual direction results from adding the current digit's angular increment to the accumulated direction from all preceding digits.

The direction is always normalized to the range from 0° up to, but not including, 360°.

## 3. How the trajectory is calculated

The algorithm follows four steps.

**Step 1 – Establish the initial direction**

The first digit determines the starting direction:

\[
\theta_0 = d_0 \cdot 36^\circ
\]

The starting position is the origin:

\[
P_0=(0,0)
\]

The first digit establishes the direction but does not produce a movement.

**Step 2 – Update the direction**

For every subsequent digit, its angular increment is added to the previous direction:

\[
\theta_n=(\theta_{n-1}+d_n\cdot36^\circ)\bmod360^\circ
\]

This means that the direction depends on the complete sequence of digits processed so far.

**Step 3 – Move one unit**

After updating the direction, the program creates a movement of exactly one unit:

\[
x_n=x_{n-1}+\cos(\theta_n)
\]

\[
y_n=y_{n-1}+\sin(\theta_n)
\]

The angle is converted to radians when used in the trigonometric functions.

**Step 4 – Repeat**

The algorithm repeats the direction update and unit movement for every remaining digit.

Every digit therefore contributes one movement, except the first digit, which only initializes the starting direction.

A digit of 0 does not change the current direction, but it still produces a movement of one unit in the direction already established.

## 4. Example

Consider the digit sequence:

**3 – 1 – 4 – 1 – 5**

The first digit establishes the initial direction at 108°.

The subsequent digits change that direction cumulatively:

| Digit | Angular increment | Resulting direction | Action |
|---:|---:|---:|---|
| 3 | 108° | 108° | Initialize direction; no movement |
| 1 | 36° | 144° | Move 1 unit |
| 4 | 144° | 288° | Move 1 unit |
| 1 | 36° | 324° | Move 1 unit |
| 5 | 180° | 144° | Move 1 unit |

Notice that the direction after processing the digit 5 is 144°, not 180°. It is the accumulated result of all angular increments.

Each new movement starts exactly where the previous movement ended.

## 5. The resulting visualisation

Every calculated position becomes a point in a two-dimensional Cartesian coordinate system. Connecting the successive positions produces the trajectory.

The visualisation provides several ways to explore the resulting path:

- **Coloured points:** Represent positions along the trajectory, with colours indicating chronological position or, optionally, the corresponding digit.
- **Coordinate system:** Provides spatial orientation through the marked axes and coordinate grid.
- **Zoom:** Allows detailed examination of smaller sections of the trajectory.
- **Pan:** Allows movement through the complete visualisation.
- **Direction indicators:** Summarise dominant angular changes in selected local sections of the digit sequence.

For very large sequences, the program uses parallel CPU processing where configured and OpenGL for rendering. Only the points required for the current view are uploaded to the graphics device.

## 6. The intention of the experiment

The program does not calculate π. It takes an existing sequence of its decimal digits and applies a consistent geometric interpretation to it.

Each digit changes the accumulated direction by a multiple of 36 degrees. Each resulting direction produces a movement of one unit.

The central question is what geometric structures emerge when the digits of π are translated into successive changes of direction and connected movements.

The trajectory provides an interactive way to explore the numerical sequence, investigate its spatial distribution and examine its appearance at different scales.

**In essence: The first digit establishes the direction. Every subsequent digit changes that direction and produces one unit of movement. The complete sequence of π becomes a continuous two-dimensional trajectory.**
