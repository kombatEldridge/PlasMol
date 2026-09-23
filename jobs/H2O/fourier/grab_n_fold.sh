#!/bin/bash

for d in x y z; do cp $d/pField.csv $d.csv; done

python column.py x.csv x xx.csv
python column.py y.csv y yy.csv
python column.py z.csv z zz.csv
python fold.py

