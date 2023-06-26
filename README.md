
<img src="/img/EC39.png" alt="EC3.9" style="height: 100px; width:100px;"/>

# N9_MIP_solver

A possible way to find the minimal optimal perturbation that change the model decision (adversarial attack) is to transform the problem, with the help of binary variables, into a Mixed Integer Program (MIP). We propose a global optimization approach to get the optimal perturbation using a dedicated branch-and-bound algorithm. 

## Requirements
GNU Gurobi's Python interface,
python3.6 or higher, tensorflow 1.11 or higher, numpy.

Install Gurobi:

```
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib
python3 setup.py install
cd ../../
```


Update environment variables:
```
export GUROBI_HOME="$PWD/gurobi912/linux64"
export PATH="$PATH:${GUROBI_HOME}/bin"
export CPATH="$CPATH:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib
```

If gurobipy is not found despite executing python setup.py install in the corresponding gurobi directory, gurobipy can alternatively be installed using conda with:
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

Install ELINA:
```
git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure -use-deeppoly -use-gurobi -use-fconv -use-cuda
cd ./gpupoly/
cmake .
cd ..
make
make install
cd ..
```

Install mpfr:
```
wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
make install
cd ..
rm mpfr-4.1.0.tar.xz
```
