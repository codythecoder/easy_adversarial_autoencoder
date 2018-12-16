from src.misc import Flip_Flopper

# Training Params
num_steps = 1000000
batch_size = 50
learning_rate = 0.0002

color_channels = 3
deconvolution_layers = 3

target_shape = 40, 56
latent_size = 150

test_freq = 100
save_freq = 100
print_freq = 1

save_count = 10

blend_steps = 3

start_layers = 512

autoencoder_target = 0.002
flip_flop = Flip_Flopper(4, (2, 2), names=('autoencoder', 'gan', 'gan-replay'))
