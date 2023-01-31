import tensorflow as tf
a = tf.data.Dataset.range(1,10)
# b = tf.data.Dataset.range(4, 7)
# ds = tf.data.Dataset.zip((a, b))
# ds = a.shuffle(2)
ds = a.repeat(count=4)
print_line = ''
for i, line in enumerate(ds):
    if i % 9 == 0:
        print_line += ' '
    print_line += str(line.numpy())

print(print_line)
