#/bin/sh
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 -m venv ./venv 
source ./venv/bin/activate
source /opt/intel/oneapi/setvars.sh
echo "*************************************************************************************"
echo "* Activate kernel parameters with ./setkernel.sh as root to allow vtune du run fine *"
echo "*************************************************************************************"
