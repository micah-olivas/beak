<div style="display: flex; align-items: center;">
     <img src="misc/icon-transparent.png" alt="BEAK icon" align="right" height=200pt/>
     <h1 style="margin: 0;">BEAK</h3>
</div>
Beak is a toolkit for modeling biophysical and evolutionary associations in proteins. It addresses several common challenges that experimentalists in protein biophysics and biochemistry face when modeling measurements in the context of evolutionary sequence data. The first is that working with large sequence datasets is too memory and compute-intensive to run in common analysis environments (e.g. a local interactive Python notebook) on a personal laptop. Beak streamlines public database queries and alignment by offloading these processes to a local server and receiving results, all within a single notebook environment.

### Installation
Beak needs to first be installed before use. It requires `python>=3.8`. You can install Beak using:
```
git clone https://github.com/micah-olivas/beak.git && cd beak

pip install -e .
```

