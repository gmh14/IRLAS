for model_name in "IRLAS"
do
  python tools/test_time.py -b 16 -n 5 -save ${model_name} --write ./caffe_prototxt/${model_name}.prototxt
done
