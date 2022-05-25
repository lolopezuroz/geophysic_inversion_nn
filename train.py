import tensorflow as tf

tf.random.set_seed(0)


def train(args,train_dataset,test_dataset):

    def print_metrics() -> str:
        """
        write a summary of metrics and reset them

        return:str
        """
        string = ""
        for output, product_metrics in metrics.items():
            string += output+" : "
            for metric in product_metrics:
                string += metric.name+" "+str(float(metric.result()))+" "
                metric.reset_states()
            string += "| "
        return string

    @tf.function
    def train_step(inputs, ground_truths) -> list:
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss_values = [losses[output_name](ground_truth, logit) * loss_weights[output_name] for output_name, ground_truth, logit in zip(losses.keys(), ground_truths, logits)]
        gradients = tape.gradient(loss_values, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for output_name, ground_truth, logit in zip(metrics.keys(), ground_truths, logits):
            for metric in metrics[output_name]:
                metric.update_state(ground_truth, logit)
        return loss_values

    @tf.function
    def test_step(inputs, ground_truths) -> list:
        logits = model(inputs, training=False)
        loss_values = [losses[output_name](ground_truth, logit) * loss_weights[output_name] for output_name, ground_truth, logit in zip(losses.keys(), ground_truths, logits)]
        for output_name, ground_truth, logit in zip(metrics.keys(), ground_truths, logits):
            for metric in metrics[output_name]:
                metric.update_state(ground_truth, logit)
        return loss_values

    # get training parameters (see parameters.py for details)
    epochs=args["epochs"]
    losses=args["losses"]
    metrics=args["metrics"]
    learning_rate=args["learning_rate"]
    optimizer=args["optimizer"]
    checkpoint_dir=args["checkpoint_dir"]
    loss_weights=args["loss_weights"]

    model = args["model"](args["inputs"],args["outputs"])

    optimizer = optimizer(learning_rate=learning_rate)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for step, sample in enumerate(train_dataset):
            inputs, ground_truths = [sample[0]], sample[1:]

            loss_values = train_step(inputs, ground_truths)
            if step % 20 == 0: print("\tTraining loss (for one batch) at step "+str(step)+" :",float(loss_values[0]), float(loss_values[1]))
        
        print("Training | "+print_metrics(metrics))
        
        for sample in test_dataset:
            inputs, ground_truths = [sample[0]], sample[1:]
            loss_values = test_step(inputs, ground_truths)

        print("Validation | "+print_metrics(metrics))

        model.save_weights(checkpoint_dir)

    return model