from CNN_Block import CNN_Block

# image channel = in_channel (1 channel - grayscale)
# output channel - kinds of kernel features extracted (vertical, horizontal, angle (45, 90, ...))
# Not pass data yet
cnn_block = CNN_Block(in_channel=3, out_channel=10)
print(cnn_block)

