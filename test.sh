for name in 'IRLAS_mobile'
do
  python -u validate.py -b 256 -j 16 --model=$name
done
