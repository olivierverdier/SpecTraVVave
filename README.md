# SpectraVVave: Compute traveling waves

[![Build Status](https://github.com/olivierverdier/SpecTraVVave/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/olivierverdier/SpecTraVVave/actions/workflows/python_package.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/olivierverdier/SpecTraVVave/graph/badge.svg?token=Ea4XsTXw6A)](https://codecov.io/github/olivierverdier/SpecTraVVave)
![Python version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg?logo=python&logoColor=gold)
[![arXiv](https://img.shields.io/badge/arXiv-1606.01465-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/1606.01465)

## Getting started

Here is a simple example to get started with SpectraVVave:

We first import the relevant bits in SpectraVVave:

```python
from travwave.diagram import BifurcationDiagram
import travwave.equations as teq
import travwave.boundary as tbc
```

Define the half length of the travelling wave:
```python
length = 30
```

Which equation are we solving. Look in the `travwave/equations` folder, or implement your own equation. Here we choose the KDV equation.
```
equation = teq.kdv.KDV(length)
```

Which boundary condition are we using? You will find some possible boundary conditions in `travwave/bounday`, or you can implement your own. Here we use the `Minimum` boundary condition, which enforces the minimum to be at zero.
```python
boundary_cond = tbc.Minimum()
```

Setup the diagram object, initialize and run it:
```python
bd = BifurcationDiagram(equation, boundary_cond)
# initialize it with default parameters
bd.initialize()
# run for fifty steps
bd.navigation.run(50)
```

Let us see what the amplitude reached is:
```
print('Amplitude = ', bd.navigation[-1]['parameter'][bd.navigation.amplitude_])
```

We plot the current computed solution, at coarse resolution:
```python
bd.plot_solution(bd.navigation[-1]['solution'])
```
![Refined solution](https://github.com/olivierverdier/SpecTraVVave/raw/main/coarse.png)

We refine to get a higher resolution travelling wave:
```python
new_size = 500
refined, v, parameter = bd.navigation.refine_at(new_size)
```

and we plot that refined solution:
```python
bd.plot_solution(refined)
```

![Refined solution](https://github.com/olivierverdier/SpecTraVVave/raw/main/wave.png)

We plot the bifurcation diagram, as well as the last refined parameter:
```python
bd.nplot_diagram()
plt.plot(parameter[0], parameter[1], 'or')
```

![Bifurcation Diagram](https://github.com/olivierverdier/SpecTraVVave/raw/main/diagram.png)
