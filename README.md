# Experimenting with VAEs latent space disentanglement
This is an implementation of a rather simplistic idea for tasks with known at train-time (and important) categorical labels: disentangled latent space is a one in which certain regions represent certain categories. For example, i would argue that to get a VAE to generate a mix between a digit 0 and a digit 8, it is important to
1. have areas of the latent space that represent digit 0 and digit 8 separately, and
2. have them share a border.

Key phrase here: "i would argue". I lack the expertise to actually reason about it in an at least semi-formal way, so i will do what great philosophers did a thousand years ago: state things and hope they make sense.

I decided to experiment with this idea because i took a look at CVAEs and noticed they usually concatenate the labels to the latent space - and i expect that there is a lot of tasks in which labels are not available during inference. I would love to use the kl divergence and encourage a different mean for each label during the training process, but i could not imagine a sensible way to both fit the means and penalize deviation from them.

The crux of the implementation is a new pathway in the model that tries to guess the label based on the latent space, and a new regularization loss to facilitate that. This loss is weighted, and the weight starts high, forcing the model to learn a latent space with good class separation, and then drops to almost zero, allowing the training process to actually work as once intended.

## Imporant notes
1. I think the iffiest decision i made is that i, to steal some importance from the mean regularization, i changed the kl divergence formula into an abomination that has an abs(means) instead of square(means). The reason for it is that i wanted to be able to drop my new classification loss to zero or almost zero without the class means shrinking into oblivion - if the decoder decodes well and regions of the latent space mean something for the classification, its good enough for me. The other option was to never drop the new classification loss completely, but that sounds like cheating as the MNIST task is so easy that i can get away without the kl divergence whatsoever.
2. I stole some code from https://keras.io/examples/generative/vae/ to make my life easier and also make sure im not making idiotic mistakes.
3. This experiment is probably done already somewhere, maybe even as a prerequisite to CVAE (feels like it, however i couldnt find no actual info), but if it inspires you to do something cool, please drop an issue! Im going through my "VAE is all you need" phase and naturally im very curious about what can be done with them
