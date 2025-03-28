Castelli & Kurucz 2004 and Phoenix models should be downloaded from 
https://archive.stsci.edu/hlsp/reference-atlases

and put in this directory.

Also, need to set `PSYN_CDBS` enivorinment variable to point to this directory. In Python, you could do this:
```
os.environ['PYSYN_CDBS'] = 'reference/trds/'
```

The expected  directory structure is 
- reference/
  - trds/
    - grid/
      - cdk04models/
      - phoenix/  