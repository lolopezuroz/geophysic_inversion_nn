from functions.importation import os, math, tensorflow as tf, numpy as np
from functions.usual_functions import unique_id, exist_directory

tf.random.set_seed(0)

def train(args, train_dataset, test_dataset):

    def print_metrics() -> str:
        """
        write a summary of metrics and reset them

        return:str the message to be printed
        """
        string = ""
        for output, product_metrics in zip(args["outputs"],metrics):
            string += f"{output} : "
            for metric in product_metrics:
                string += f"{metric.name} {float(metric.result())} "
                metric.reset_states()
            string += "| "
        return string

    @tf.function
    def train_step(inputs, ground_truths) -> list:
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss_values = [loss(ground_truth, logit) * loss_weight for loss, loss_weight, ground_truth, logit in zip(losses, loss_weights, ground_truths, logits)]
            gradients = tape.gradient(loss_values, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for i, (ground_truth, logit) in enumerate(zip(ground_truths, logits)):
            for metric in metrics[i]:
                metric.update_state(ground_truth, logit)
        return loss_values

    @tf.function
    def test_step(inputs, ground_truths) -> list:
        logits = model(inputs, training=False)
        loss_values = [loss(ground_truth, logit) * loss_weight for loss, loss_weight, ground_truth, logit in zip(losses, loss_weights, ground_truths, logits)]
        for i, (ground_truth, logit) in enumerate(zip(ground_truths, logits)):
            for metric in metrics[i]:
                metric.update_state(ground_truth, logit)
        return loss_values

    # get training parameters (see training_parameters.py for details)
    epochs=args["epochs"]
    losses=args["losses"]
    metrics=args["metrics"]
    learning_rate=args["learning_rate"]
    optimizer=args["optimizer"]
    loss_weights=args["loss_weights"]

    model = args["model"](args["early_fusion"], args["inputs"], args["outputs"])

    n_input = len(args["inputs"])

    optimizer = optimizer(learning_rate=learning_rate)

    checkpoint_dir = os.path.join("./train_states", unique_id())
    checkpoint_dir_best = os.path.join(checkpoint_dir, "best")
    exist_directory(checkpoint_dir_best)
    
    min_error = math.inf # initialize minimal error at infinity

    for epoch in range(epochs):

        checkpoint_dir_i_epoch = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        exist_directory(checkpoint_dir_i_epoch)

        print(f"\nStart of epoch {epoch}")
        for step, sample in enumerate(train_dataset):
            inputs, ground_truths = sample[0:n_input], sample[n_input:]

            loss_values = np.nan_to_num(train_step(inputs, ground_truths))
            if step % 20 == 0: print(f"\tTraining loss (for one batch) at step {step} :", float(loss_values[0]), float(loss_values[1]))
        
        print(f"Training | {print_metrics()}")
        
        for sample in test_dataset:

            inputs, ground_truths = sample[0:n_input], sample[n_input:]
            loss_values = np.nan_to_num(test_step(inputs, ground_truths))
            error = float(loss_values[-1])

        print(f"Validation | {print_metrics()}")

        model.save_weights(checkpoint_dir_i_epoch)
        if min_error > error:
            min_error = error
            model.save_weights(checkpoint_dir_best)

    return model