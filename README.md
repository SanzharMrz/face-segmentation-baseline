# face-segmentation-baseline

Face segmentation baseline

# Install

```bash
# Create a venv
$ python -m virtualenv venv
$ source venv/bin/activate

# Install requirements
$ pip install -r requirements.txt
```

# Usage
```bash
$  python inference.py --filepath Aishwariya_Rai_\(face\).jpg 
```

# Results
After running scripts you will have, two files, for examples __array__ file, which you can load:

```python
import numpy as np
a = np.loadtxt('Aishwariya_Rai_(face)_array.out', delimiter=',')
```

And __parsed__ file, with segmentation results:

![Aishwariya_Rai_(face)_parsed.png](https://user-images.githubusercontent.com/48170101/131503009-fadacf99-4896-4686-9457-fa74328b84ba.png)

