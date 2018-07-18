OUTPUT_SO=cw2vec_ops.so
[ -f $OUTPUT_SO ] && rm $OUTPUT_SO
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared cw2vec_ops.cc cw2vec_kernels.cc -o $OUTPUT_SO -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
file $OUTPUT_SO
