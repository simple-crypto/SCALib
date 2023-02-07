---
name: Bug report
about: Tell us when SCALib crashes or provides wrong results.
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**

Please provide a simple script that reproduces the bug. If this needs data, please minimize the dataset size. Also, if possible, generate the data using pseudo-radomness (e.g., `numpy.random`).

In any case, please set fixed seeds using
```py
numpy.random.seed(0)
random.seed(0)
```

**Observed behavior**

```
Paste the output of the script here.
```

Provide any additional comment on the problematic behavior (running time, RAM or CPU usage...).

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment (please complete the following information):**
 - OS: [e.g. ubuntu 22.04]
- Python version: [output of `<python> -V` where `<python>` is how your run Python]
- numpy version: [output of `<python> -c "import numpy; print(numpy.__version__)"`]
- SCALib version: [output of `<python> -c "import scalib; print(scalib.__version__)"`]
- How did you install SCALib? [fom PyPI or did you build it yourself?]
- What CPU do you use? [intel, amd, ARM...]

**Additional context**
Add any other context about the problem or particularities about your configuration (e.g., environment variables, python install, etc.).
