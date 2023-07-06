import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pandas as pd

def plot_latent(decoder, scale, nonlinear=False):
    # display a n * n 2D manifold of images
    n = 25
    img_dim = 28
    if nonlinear:
        scale = np.sqrt(scale)
    figsize = 15
    figure = np.zeros((img_dim * n, img_dim * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    if nonlinear:
        grid_x = np.square(np.abs(grid_x))*np.sign(grid_x)
        grid_y = np.square(np.abs(grid_y))*np.sign(grid_y)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder(z_sample, training=False).numpy()
            images = x_decoded[0].reshape(img_dim, img_dim)
            figure[
                i * img_dim : (i + 1) * img_dim,
                j * img_dim : (j + 1) * img_dim,
            ] = images

    plt.figure(figsize =(figsize, figsize))
    start_range = img_dim // 2
    end_range = n * img_dim + start_range
    pixel_range = np.arange(start_range, end_range, img_dim)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap ="Greys_r")
    plt.show()


def evaluate(model, x_test, y_test):
    intermediate_layer_model = keras.Model(inputs=model.encoder.input, outputs=model.encoder.get_layer('z_log_var').output)
    variances = np.exp(0.5*intermediate_layer_model.predict(x_test[[0],:,:,:]))
    print(f'{variances = }')
    pred_means, pred_vars = model.encoder.predict(x_test)
    preddf = pd.DataFrame(pred_means, columns=['x','y']).assign(label=y_test)

    plt.scatter(
        preddf['x'],
        preddf['y'],
        c=preddf['label'],
        s=1,
        alpha=0.3,
    )

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(x_test[28, ..., 0])
    axs[1].imshow(model.predict(x_test[[28]])[0, ..., 0])

    plot_latent(
        model.decoder,
        preddf[['x','y']].abs().quantile(0.9).max()
    )
